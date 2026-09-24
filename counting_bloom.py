"""
counting_bloom.py — 单文件分层计数布隆过滤器（仅标准库）

场景：海量 URL 去重，集合持续增长，需要 add / contains / remove，
删除不能误伤其他元素（计数器方案天然满足：每个槽位存计数而非比特，
删除只做 -1，不会把别的元素"顶掉"——这与布隆过滤器的 bit 置 0 方案不同）。

====================================================================
取舍一：容量扩容策略 —— 整体重建 vs 分层结构
====================================================================
方案 A：整体重建（rehash / resize）
  - 做法：容量不足时分配更大的位数组，把所有历史元素重新插入。
  - 优点：结构简单，任意时刻只有一张表，内存利用率最高，FPR 精确可控。
  - 缺点：① 需要拿到全量原始数据才能重建（布隆过滤器本身不可逆，
    线上往往已丢原始 URL）；② 重建期间要么停服要么双写，一次性
    CPU/内存峰值高。
  - 适用：离线/批处理场景、能容忍分钟级暂停、或原始数据可回放
    （如 Kafka 可重放）的系统。

方案 B：分层结构（Scalable Bloom Filter，本实现采用）
  - 做法：当前层写满后冻结为只读，新建更大的一层承接新插入；
    查询跨所有层取 OR；删除从最新层向最旧层逐层查找并递减。
  - 优点：扩容 O(1) 摊销、无停服、不需要原始数据，天然适配
    "集合持续变大"的线上场景。
  - 缺点：① 查询/删除要跨层（层数 ~log_growth(总量/初始容量)，
    增长 1000 倍也只有 ~10 层）；② 总 FPR 是各层之和的上界
    （通过 tighten 因子让新层 FPR 几何衰减来约束，见下）；
    ③ 删除必须跨层定位元素所在层。
  - 适用：7x24 在线服务、数据流不可回放、写入持续增长的场景
    —— 即本题场景，故选 B。

====================================================================
取舍二：假阳性率 vs 内存的权衡公式与默认参数推导
====================================================================
经典结论（单层，n = 预期元素数，p = 目标假阳性率）：
    最优槽位数  m = -n * ln(p) / (ln 2)^2          （槽）
    最优哈希数  k = (m / n) * ln 2 = -log2(p)      （个）
计数布隆过滤器每槽用 c 位计数器，故单层内存 = m * c 位。

默认参数推导（initial_capacity=1_000_000, error_rate=0.001, counter_bits=4）：
    m = -1e6 * ln(1e-3) / 0.4804 ≈ 14,377,759 槽
    k = round(14.38 * 0.693)   = 10
    单层内存 = 14.38M 槽 * 4 bit ≈ 6.9 MB（每元素约 57.6 bit）
即：p 每缩小 10 倍，每元素多花 ~4.8 bit（ln10/(ln2)^2 * c），
这就是"FPR 一个数量级 ↔ 固定比特数"的交换比。

分层总 FPR：第 i 层目标 p_i = p * tighten^i（tighten=0.5），
总 FPR ≤ Σ p_i = p / (1 - tighten) = 2p，有确定上界。
新层容量 = 上一层 * growth（默认 2 倍几何增长，摊销 O(1)）。

====================================================================
计数器溢出策略（同一槽被插太多次）
====================================================================
每槽计数器上限 MAX = 2^c - 1（c=4 时 MAX=15）。
由上面推导，每槽计数均值 = k*n/m = ln2 ≈ 0.693，服从泊松分布，
P(单槽 ≥ 16) < 1e-15 —— 正常参数下溢出几乎不可能，但仍需定义行为：
  - overflow="saturate"（默认）：到达 MAX 后不再递增（钉住），
    对应删除在 MAX 处也不再递减。宁可让该槽"粘"在 1（可能遗留
    假阳性），也绝不回绕到 0 造成假阴性（去重场景假阴性=漏判
    已访问 URL，比假阳性危害大）。
  - overflow="error"：递增会溢出时抛 OverflowError，由调用方
    决定（如触发告警或人工扩容），适合强一致审计场景。
注意：计数器语义是多重集——同一元素重复 add 会重复计数，
remove 只减一次。若要去重语义，调用方应先 contains 再 add。

删除语义：remove 从最新层向最旧层找第一个 contains 该元素的层，
将其 k 个槽各减 1。只对"确实 add 过"的元素调用 remove；
对假阳性元素调用 remove 会误减别人的计数（所有计数布隆过滤器
的固有约束），本实现找不到时抛 KeyError 作为保护。
"""

from __future__ import annotations

import hashlib
import math

__all__ = ["CountingBloomFilter", "optimal_num_slots", "optimal_num_hashes"]

_LN2 = math.log(2.0)


def optimal_num_slots(capacity: int, error_rate: float) -> int:
    """m = -n * ln(p) / (ln2)^2"""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must be in (0, 1)")
    return max(8, math.ceil(-capacity * math.log(error_rate) / (_LN2 * _LN2)))


def optimal_num_hashes(num_slots: int, capacity: int) -> int:
    """k = (m / n) * ln2"""
    return max(1, round(num_slots / capacity * _LN2))


def _hash_positions(item, k: int, m: int):
    """双重哈希派生 k 个位置：g_i = (h1 + i*h2) mod m。

    基础哈希用 blake2b-128（标准库、跨进程确定性，不像内置 hash()
    受 PYTHONHASHSEED 影响），拆成两个 64 位值后按
    Kirsch-Mitzenmacher 方法线性派生，理论证明与 k 个独立哈希
    的 FPR 渐近等价。h2 强制为奇数，保证 i*h2 能遍历更多槽位。
    """
    data = item if isinstance(item, (bytes, bytearray)) else str(item).encode("utf-8")
    digest = hashlib.blake2b(bytes(data), digest_size=16).digest()
    h1 = int.from_bytes(digest[:8], "little")
    h2 = int.from_bytes(digest[8:], "little") | 1
    for i in range(k):
        yield (h1 + i * h2) % m


class _Layer:
    """单层计数布隆过滤器。写满后冻结为只读（删除仍允许）。"""

    __slots__ = ("capacity", "error_rate", "m", "k", "counter_bits",
                 "max_counter", "counters", "count")

    def __init__(self, capacity: int, error_rate: float, counter_bits: int = 4):
        if counter_bits not in (4, 8):
            raise ValueError("counter_bits must be 4 or 8")
        self.capacity = capacity
        self.error_rate = error_rate
        self.m = optimal_num_slots(capacity, error_rate)
        self.k = optimal_num_hashes(self.m, capacity)
        self.counter_bits = counter_bits
        self.max_counter = (1 << counter_bits) - 1
        # 4 bit: 每字节装 2 个计数器（低半字节在前）；8 bit: 每槽一字节
        nbytes = (self.m + 1) // 2 if counter_bits == 4 else self.m
        self.counters = bytearray(nbytes)
        self.count = 0  # 本层净插入次数（add - remove）

    @property
    def full(self) -> bool:
        return self.count >= self.capacity

    @property
    def memory_bytes(self) -> int:
        return len(self.counters)

    def _get(self, i: int) -> int:
        if self.counter_bits == 8:
            return self.counters[i]
        b = self.counters[i >> 1]
        return (b >> 4) if (i & 1) else (b & 0x0F)

    def _set(self, i: int, v: int) -> None:
        if self.counter_bits == 8:
            self.counters[i] = v
            return
        j = i >> 1
        if i & 1:
            self.counters[j] = (self.counters[j] & 0x0F) | (v << 4)
        else:
            self.counters[j] = (self.counters[j] & 0xF0) | v

    def might_contain(self, item) -> bool:
        return all(self._get(p) > 0 for p in _hash_positions(item, self.k, self.m))

    def add(self, item, overflow: str) -> None:
        for p in _hash_positions(item, self.k, self.m):
            c = self._get(p)
            if c == self.max_counter:
                if overflow == "error":
                    raise OverflowError(
                        f"counter overflow at slot {p} (max={self.max_counter})"
                    )
                continue  # saturate: 钉在 MAX，不回绕
            self._set(p, c + 1)
        self.count += 1

    def discard(self, item, overflow: str) -> None:
        """前提：might_contain(item) 为真且该元素确实插入过本层。"""
        for p in _hash_positions(item, self.k, self.m):
            c = self._get(p)
            if c == 0:
                # 理论上不该发生（might_contain 已保证全 >0），防御性处理
                raise KeyError(f"inconsistent state: slot {p} is 0 during remove")
            if c == self.max_counter and overflow == "saturate":
                continue  # 饱和槽无法知道真实计数，保持钉住
            self._set(p, c - 1)
        self.count -= 1


class CountingBloomFilter:
    """分层计数布隆过滤器：旧层只读，新层承载写入，查询/删除跨层。"""

    def __init__(
        self,
        initial_capacity: int = 1_000_000,
        error_rate: float = 0.001,
        growth: float = 2.0,
        tighten: float = 0.5,
        counter_bits: int = 4,
        overflow: str = "saturate",
    ):
        if growth <= 1.0:
            raise ValueError("growth must be > 1")
        if not 0.0 < tighten < 1.0:
            raise ValueError("tighten must be in (0, 1)")
        if overflow not in ("saturate", "error"):
            raise ValueError("overflow must be 'saturate' or 'error'")
        self.growth = growth
        self.tighten = tighten
        self.counter_bits = counter_bits
        self.overflow = overflow
        self.layers = [_Layer(initial_capacity, error_rate, counter_bits)]

    def add(self, item) -> None:
        layer = self.layers[-1]
        if layer.full:
            new_capacity = math.ceil(layer.capacity * self.growth)
            new_error = layer.error_rate * self.tighten
            self.layers.append(_Layer(new_capacity, new_error, self.counter_bits))
            layer = self.layers[-1]
        layer.add(item, self.overflow)

    def contains(self, item) -> bool:
        return any(layer.might_contain(item) for layer in self.layers)

    def remove(self, item) -> None:
        """从最新层向最旧层定位并递减一次。元素不存在时抛 KeyError。"""
        for layer in reversed(self.layers):
            if layer.might_contain(item):
                layer.discard(item, self.overflow)
                return
        raise KeyError(f"item not present: {item!r}")

    @property
    def count(self) -> int:
        """净插入次数（多重集语义，重复 add 会重复计）。"""
        return sum(layer.count for layer in self.layers)

    @property
    def memory_bytes(self) -> int:
        return sum(layer.memory_bytes for layer in self.layers)

    def fpr_upper_bound(self) -> float:
        """总假阳性率上界：1 - Π(1 - p_i) ≤ Σ p_i。"""
        return 1.0 - math.prod(1.0 - layer.error_rate for layer in self.layers)

    def info(self) -> str:
        lines = [
            f"layers={len(self.layers)} count={self.count} "
            f"memory={self.memory_bytes / 1024 / 1024:.2f}MiB "
            f"fpr_upper_bound={self.fpr_upper_bound():.2e}"
        ]
        for i, layer in enumerate(self.layers):
            lines.append(
                f"  layer{i}: capacity={layer.capacity} count={layer.count} "
                f"m={layer.m} k={layer.k} p={layer.error_rate:.2e} "
                f"{'readonly' if layer.full else 'active'}"
            )
        return "\n".join(lines)

    def __contains__(self, item) -> bool:
        return self.contains(item)

    def __len__(self) -> int:
        return self.count


if __name__ == "__main__":
    import random
    import string

    random.seed(42)

    def rand_url() -> str:
        path = "".join(random.choices(string.ascii_lowercase, k=12))
        return f"https://example.com/{path}"

    print("=== 1. 基本 add / contains / remove ===")
    bf = CountingBloomFilter(initial_capacity=10_000, error_rate=0.001)
    urls = [rand_url() for _ in range(2000)]
    for u in urls:
        bf.add(u)
    print("全部插入后 contains 命中:",
          all(bf.contains(u) for u in urls))  # 必为 True（无假阴性）

    victim = urls[0]
    bf.remove(victim)
    print("remove 后目标元素 contains:", bf.contains(victim))  # False
    rest_ok = all(bf.contains(u) for u in urls[1:])
    print("remove 后其余元素不受影响:", rest_ok)  # True：删除不顶掉别人

    print("\n=== 2. 同一元素多次插入/删除（计数器语义）===")
    bf.add("https://dup.example/")
    bf.add("https://dup.example/")
    bf.remove("https://dup.example/")
    print("删一次后仍在（还剩一次计数）:", bf.contains("https://dup.example/"))
    bf.remove("https://dup.example/")
    print("删两次后消失:", not bf.contains("https://dup.example/"))

    print("\n=== 3. 删除不存在的元素有保护 ===")
    try:
        bf.remove("https://never-seen.example/")
    except KeyError as e:
        print("抛出 KeyError:", e)

    print("\n=== 4. 持续写入触发分层扩容（初始容量仅 1 万）===")
    for _ in range(60_000):
        bf.add(rand_url())
    print(bf.info())

    print("\n=== 5. 实测假阳性率（用未插入的样本）===")
    false_pos = sum(1 for _ in range(20_000) if bf.contains(rand_url()))
    empirical = false_pos / 20_000
    print(f"经验 FPR = {empirical:.4%}，理论上界 = {bf.fpr_upper_bound():.4%}")
    assert empirical <= bf.fpr_upper_bound(), "经验 FPR 超出理论上界！"

    print("\n=== 6. 计数器溢出策略演示（error 模式）===")
    tiny = CountingBloomFilter(initial_capacity=8, error_rate=0.5,
                               counter_bits=4, overflow="error")
    try:
        for i in range(200):
            tiny.add("hot-key")  # 同一元素反复插，计数器必然顶到 15
    except OverflowError as e:
        print("按定义抛出 OverflowError:", e)
    saturated = CountingBloomFilter(initial_capacity=8, error_rate=0.5)
    for i in range(200):
        saturated.add("hot-key")  # saturate 模式：钉在 15，不报错不回绕
    print("saturate 模式下 hot-key 仍 contains:", saturated.contains("hot-key"))

    print("\n全部示例断言通过 ✔")
