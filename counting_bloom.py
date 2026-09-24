# -*- coding: utf-8 -*-
"""
counting_bloom.py — 单文件、可扩容的计数布隆过滤器（仅依赖 Python 标准库）

适用场景：海量 URL 去重（是否访问过），集合持续增长，初始位数组很快装不下。

=====================================================================
 关键取舍一：容量扩容策略 —— 本实现选择【分层结构】
=====================================================================
方案 A：整体重建（rehash）
  - 做法：申请更大的位数组，把全部已存元素重新哈希写入。
  - 优点：结构简单、单层查询快、内存利用率最高。
  - 缺点：一次性成本高（O(n) 全量重写）；重建期间要么停服、要么双写；
          且布隆过滤器本身无法枚举已存元素，重建必须依赖外部重新喂数据。
  - 适用：数据有外部持久化来源可重放、允许短暂停服或双写窗口的场景
          （如离线重建 + 定时切换）。

方案 B：分层结构（layered / scalable，本实现采用）
  - 做法：当前层写满后将其冻结为只读，新建一个容量更大、目标误判率更严
          的新层承载后续插入；查询时跨所有层 OR；删除时从新到旧逐层定位。
  - 优点：扩容是 O(1) 的（只分配新层），全程无停服、无数据搬迁；
          不依赖外部数据源重放。
  - 缺点：查询/删除要跨层（层数 ~ log(总插入量/首层容量)，实际只有个位数）；
          内存略冗余（旧层即使删除变稀疏也不回收，除非整体重建）。
  - 适用：在线服务、集合只增不减或持续增长、不能停机的场景 ——
          正是题目描述的线上 URL 去重场景，故选 B。

跨层删除的歧义问题（必须显式处理）：
  元素只真实存在于某一个层，但其它层可能因假阳性也"命中"它。
  若盲目在第一个命中的层减计数，可能选错层，把该层其它元素共享的
  计数槽清零，造成假阴性（把别的元素"顶掉"）—— 这是分层 CBF 删除
  唯一会伤及其它元素的路径。本实现的策略：
    * 恰好一个层命中：该层必为真层（真层必然命中），删除精确无害；
    * 多个层同时命中：无法分辨真层，拒绝删除并返回 None（概率约等于
      单层误判率，~1% 量级）。宁可本次删不掉（元素残留为假阳性），
      也绝不冒误伤其它元素的风险；
    * 没有任何层命中：元素不存在，返回 False。
  由此保证：remove 永远不会让其它已插入元素产生假阴性。

各层目标误判率按几何级数收紧：p_i = p0 * tightening^i，
由并集界，整体误判率 <= sum(p_i) = p0 / (1 - tightening)，
因此无论涨多少层，总误判率都有确定上界。

=====================================================================
 关键取舍二：假阳性率 vs 内存 —— 公式与默认参数推导
=====================================================================
设单层预期容纳 n 个元素、m 个计数器槽、k 个哈希函数，则经典结论：

    假阳性率  p ≈ (1 - e^(-k*n/m))^k
    最优 k    k* = (m/n) * ln 2
    所需槽数  m  = -n * ln p / (ln 2)^2 ≈ 1.44 * n * log2(1/p)

默认参数推导（initial_capacity=4096, error_rate=0.01）：
    p = 1%  →  m/n = -ln(0.01)/(ln2)^2 ≈ 9.59 个槽/元素
             →  k* = 9.59 * ln2 ≈ 6.6，取 k = 7
    每槽 1 字节计数器 → 每元素约 9.6 字节内存。
    若 p 降到 0.1%，m/n ≈ 14.4（k=10），内存 +50%，误判率降 10 倍。
    规律：误判率每降 10 倍，内存约增加 4.8 字节/元素（ln10/(ln2)^2）。

=====================================================================
 计数器溢出策略：饱和（saturating counter）
=====================================================================
每槽用 1 字节计数器（0..255）。同一槽被插入第 256 次时不再增加，
保持 255（饱和），并累计 overflow_events 指标。
  - 为什么不回绕：回绕到 0 会把同槽其它元素的"已存在"信息清零，
    导致这些元素查询时假阴性 —— 不可接受。
  - 为什么删除时不减饱和槽：槽已饱和说明真实计数 >= 255，确切值未知，
    贸然减一可能把仍被其它元素占用的槽清零。故对饱和槽删除时保持不动
    （保守策略：宁可留下假阳性，绝不制造假阴性）。
  - 溢出概率：在 m/n ≈ 9.6、k=7 的默认配置下，单槽计数服从近似
    Poisson(λ=k*n/m≈0.7)，P(>=255) 实际为 0，工程上无需担心。
    若要省内存可改 4 bit 计数器（内存减半），溢出概率仍约 1e-4 量级，
    且本实现的饱和语义同样适用。
=====================================================================
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterator, List, Tuple, Union

__all__ = ["ScalableCountingBloomFilter"]

COUNTER_MAX = 255  # 单字节计数器上限（饱和值）

Item = Union[str, bytes]


def _hash_pair(item: Item) -> Tuple[int, int]:
    """用 blake2b 生成 128 bit 摘要，拆成两个 64 bit 基础哈希。

    哈希族采用双重哈希（double hashing, Kirsch-Mitzenmacher 2006）：
        h_i(x) = (h1 + i * h2) mod m,  i = 0..k-1
    只需一次真实哈希运算即可派生任意多个近似独立的哈希函数，
    理论证明其与完全独立哈希的误判率渐近等价。
    h2 强制为奇数，保证步长与 m 互质的概率高、探测序列周期长。
    """
    data = item if isinstance(item, bytes) else item.encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=16).digest()
    h1 = int.from_bytes(digest[:8], "little")
    h2 = int.from_bytes(digest[8:], "little") | 1
    return h1, h2


def _optimal_m(n: int, p: float) -> int:
    """给定预期元素数 n 和目标误判率 p，返回所需槽数 m。"""
    return max(64, math.ceil(-n * math.log(p) / (math.log(2) ** 2)))


def _optimal_k(m: int, n: int) -> int:
    """给定 m、n，返回最优哈希函数个数 k = (m/n) * ln2。"""
    return max(1, round((m / n) * math.log(2)))


class _Layer:
    """单层计数布隆过滤器：bytearray 计数器 + 双重哈希。"""

    __slots__ = ("m", "k", "capacity", "target_p", "counters", "count", "overflow_events")

    def __init__(self, capacity: int, target_p: float) -> None:
        self.capacity = capacity          # 设计容纳元素数（写满即冻结）
        self.target_p = target_p          # 本层目标误判率
        self.m = _optimal_m(capacity, target_p)
        self.k = _optimal_k(self.m, capacity)
        self.counters = bytearray(self.m)
        self.count = 0                    # 本层净增元素数（add - remove）
        self.overflow_events = 0          # 计数器饱和次数（监控指标）

    def _indices(self, item: Item) -> Iterator[int]:
        h1, h2 = _hash_pair(item)
        m = self.m
        for i in range(self.k):
            yield (h1 + i * h2) % m

    def add(self, item: Item) -> None:
        for idx in self._indices(item):
            c = self.counters[idx]
            if c < COUNTER_MAX:
                self.counters[idx] = c + 1
            else:
                # 饱和：不回绕、不再增加，只记录指标（见模块 docstring）
                self.overflow_events += 1
        self.count += 1

    def contains(self, item: Item) -> bool:
        return all(self.counters[idx] for idx in self._indices(item))

    def remove(self, item: Item) -> None:
        """调用前必须确认 contains(item) 为真。"""
        for idx in self._indices(item):
            c = self.counters[idx]
            if 0 < c < COUNTER_MAX:
                self.counters[idx] = c - 1
            # c == COUNTER_MAX（曾饱和）：真实计数未知，保守不动，
            # 避免误清其它元素占用的槽（宁可假阳性，不要假阴性）。
        self.count -= 1


class ScalableCountingBloomFilter:
    """分层可扩容计数布隆过滤器。

    参数：
        initial_capacity: 首层设计容量（写满后扩容）
        error_rate:       首层目标误判率 p0；整体误判率上界为
                          p0 / (1 - tightening)
        growth_factor:    每层容量相对上一层的倍数（默认 2）
        tightening:       每层目标误判率的收紧系数（默认 0.9，
                          保证总误判率收敛到 <= p0/0.1 = 10*p0）
    """

    def __init__(
        self,
        initial_capacity: int = 4096,
        error_rate: float = 0.01,
        growth_factor: int = 2,
        tightening: float = 0.9,
    ) -> None:
        if not (0 < error_rate < 1):
            raise ValueError("error_rate 必须在 (0, 1) 之间")
        if initial_capacity < 1 or growth_factor < 2 or not (0 < tightening < 1):
            raise ValueError("非法的容量/增长参数")
        self._growth = growth_factor
        self._tightening = tightening
        self._layers: List[_Layer] = [_Layer(initial_capacity, error_rate)]

    # ---- 核心接口 -------------------------------------------------

    def add(self, item: Item) -> None:
        """插入元素。同一元素可重复插入（计数语义），需配对 remove。"""
        top = self._layers[-1]
        if top.count >= top.capacity:
            self._grow()
            top = self._layers[-1]
        top.add(item)

    def contains(self, item: Item) -> bool:
        """查询：任一层命中即视为存在（可能假阳性，绝不假阴性）。"""
        return any(layer.contains(item) for layer in self._layers)

    def remove(self, item: Item) -> bool:
        """删除一次插入。返回值三态：
            True  —— 已精确定位到唯一所在层并删除；
            False —— 元素不存在（所有层都未命中）；
            None  —— 多个层同时命中（真层 + 其它层假阳性），无法确定
                     真层，为保护其它元素本次拒绝删除（概率 ~1% 量级，
                     见模块 docstring"跨层删除的歧义问题"）。
        注意：只能删除确认 add 过的元素；对未插入元素调用 remove 可能
        因假阳性误减别人的计数（所有 CBF 的固有限制）。
        """
        candidates = [layer for layer in self._layers if layer.contains(item)]
        if not candidates:
            return False
        if len(candidates) > 1:
            return None
        candidates[0].remove(item)
        return True

    __contains__ = contains

    # ---- 扩容 -----------------------------------------------------

    def _grow(self) -> None:
        """冻结当前顶层，新建容量 ×growth、误判率 ×tightening 的新层。"""
        prev = self._layers[-1]
        self._layers.append(
            _Layer(prev.capacity * self._growth, prev.target_p * self._tightening)
        )

    # ---- 观测 -----------------------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def __len__(self) -> int:
        """当前净元素数（重复插入会重复计数）。"""
        return sum(layer.count for layer in self._layers)

    @property
    def memory_bytes(self) -> int:
        """计数器占用的总字节数。"""
        return sum(layer.m for layer in self._layers)

    @property
    def error_rate_bound(self) -> float:
        """当前整体误判率的理论上界（各层 p_i 求和，并集界）。"""
        return sum(layer.target_p for layer in self._layers)

    @property
    def overflow_events(self) -> int:
        return sum(layer.overflow_events for layer in self._layers)

    def stats(self) -> str:
        lines = [
            f"layers={len(self._layers)}  items={len(self)}  "
            f"memory={self.memory_bytes / 1024:.1f} KiB  "
            f"fpr_bound={self.error_rate_bound:.4%}  "
            f"overflow_events={self.overflow_events}"
        ]
        for i, layer in enumerate(self._layers):
            lines.append(
                f"  layer{i}: cap={layer.capacity} m={layer.m} k={layer.k} "
                f"p={layer.target_p:.5f} count={layer.count} "
                f"load={layer.count / layer.capacity:.1%}"
            )
        return "\n".join(lines)


# =====================================================================
# 示例调用：python counting_bloom.py
# =====================================================================
if __name__ == "__main__":
    import random

    random.seed(42)

    # 首层故意调小，方便演示自动分层扩容；线上可用默认 4096 起步
    cbf = ScalableCountingBloomFilter(initial_capacity=2000, error_rate=0.01)

    # 1) 插入 10000 个 URL（会触发多次自动扩容）
    urls = [f"https://example.com/page/{i}" for i in range(10000)]
    for u in urls:
        cbf.add(u)

    # 2) 已插入的必须全部命中（布隆过滤器无假阴性）
    misses = sum(1 for u in urls if u not in cbf)
    print(f"[1] 已插入 10000 条，漏判（假阴性）数量 = {misses}  (应为 0)")

    # 3) 删除 3000 条，验证删除后不再命中，且不影响其它元素
    removed = urls[:3000]
    kept = urls[3000:]
    ok, ambiguous = 0, 0
    for u in removed:
        r = cbf.remove(u)
        assert r is not False, f"已插入的元素 remove 不应返回 False: {u}"
        if r is True:
            ok += 1
        else:
            ambiguous += 1  # 跨层歧义，本次拒绝删除（保护其它元素）
    print(f"[2] 删除 3000 条：成功 {ok}，因跨层歧义保留 {ambiguous} "
          f"(~1% 量级属正常，残留仅表现为假阳性)")
    still_there = sum(1 for u in removed if u in cbf)
    print(f"    已删除集合中仍命中 {still_there} 条 (歧义保留 + 假阳性)")
    alive_misses = sum(1 for u in kept if u not in cbf)
    print(f"    保留的 7000 条漏判 = {alive_misses}  "
          f"(必须为 0：删除绝不会顶掉其它元素)")

    # 4) 实测假阳性率：用 50000 个从未插入的 URL 探测
    probes = [f"https://other-site.com/{i}" for i in range(50000)]
    fp = sum(1 for u in probes if u in cbf)
    print(f"[3] 实测假阳性率 = {fp / len(probes):.4%}  "
          f"(理论上界 {cbf.error_rate_bound:.4%})")

    # 5) 删除不存在的元素应安全返回 False
    print(f"[4] remove(未插入元素) -> {cbf.remove('https://never-seen.com')}")

    # 6) 结构状态
    print("[5] 过滤器状态：")
    print(cbf.stats())
