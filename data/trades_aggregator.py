from dataclasses import dataclass
from typing import Literal, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import torch
import os

AggName = Literal["amount_sum", "vwap"]
EmptyPolicy = Literal["nan", "zero", "ffill"]

@dataclass
class TradesAggregator:
    """
    当前版本comments掉了缓存功能
    当前版本的因子计算逻辑是 滚动窗口计算, 默认一次滚动一个snapshot时间步长
    未来可以考虑增加 step 参数，支持更大步长的滚动计算 (通过out[k::step] 实现)
    """
    # 两个时间轴
    snapshot_ts: np.ndarray      # shape: [T], sorted
    trades_ts: np.ndarray        # shape: [N], sorted

    # 逐笔中需要合并的columns
    trades_price: np.ndarray        # shape: [N]
    trades_amount: np.ndarray       # shape: [N]

    # 时间轴的最小粒度（snapshot 之间的固定间隔）
    snapshot_step_ms: int
    device: torch.device = torch.device("cpu")
    cache_dir: Optional[str] = None
    cache_key: Optional[str] = None

    def __post_init__(self):
        # 到 snapshot i 为止，一共发生了多少条 trade
        self._right = np.searchsorted(
            self.trades_ts, self.snapshot_ts, side="right"
        )

        # 预计算 逐笔的 amount 和 price*amount 的和，加速后面的计算
        q = self.trades_amount.astype(np.float64)
        pq = (self.trades_price * self.trades_amount).astype(np.float64)

        self._cum_q = np.zeros(q.shape[0] + 1, dtype=np.float64)
        self._cum_pq = np.zeros(pq.shape[0] + 1, dtype=np.float64)
        self._cum_q[1:] = np.cumsum(q)
        self._cum_pq[1:] = np.cumsum(pq)

    def _check_lookback(self, lookback_ms: int) -> None:
        """
        检查 lookback_ms 必须是正数且是 snapshot_step_ms 的整数倍
        """
        if lookback_ms <= 0:
            raise ValueError("lookback_ms must be > 0")

        if lookback_ms % self.snapshot_step_ms != 0:
            raise ValueError(
                f"lookback_ms={lookback_ms} must be a multiple of snapshot_step_ms={self.snapshot_step_ms}"
            )
    
    def _compute_sums_for_lookback(self, lookback_ms: int):
        """
        用已经初始化的self._right 和 已经预计算的self._cum_q/_cum_pq 来计算每个 snapshot 窗口内的 sum_q 和 sum_pq
        注意： 
            1. output 从 index 为 [k:]来保证从第k个开始
            2. 现在是滚动版本，如果要调整滚动幅度或者resample则需要 [k::w], 
                w 为 rolling winow size 
        """
        k = lookback_ms // self.snapshot_step_ms
        right = self._right

        # 小于k 的窗口是0，大于等于k 的窗口 向左移动k个单位
        left = np.zeros_like(right)
        
        left[:k] = 0
        left[k:] = right[:-k]

        sum_q  = (self._cum_q[right]  - self._cum_q[left]).astype(np.float32)
        sum_pq = (self._cum_pq[right] - self._cum_pq[left]).astype(np.float32)
        return sum_q, sum_pq
    
    def _compute_tensor(
        self,
        name: AggName,
        lookback_ms: int,
        empty_policy: EmptyPolicy,
    ) -> torch.Tensor:
        """
        计算部分：根据 name 和 lookback_ms 计算聚合特征，返回 shape: [T,1] 的 tensor
        """
        T = self.snapshot_ts.shape[0]
        k = lookback_ms // self.snapshot_step_ms

        # step 1: 基础窗口统计量
        sum_q, sum_pq = self._compute_sums_for_lookback(lookback_ms)

        out = np.full(T, np.nan, dtype=np.float32)
        # step 2: 根据 name 做相应计算
        if name == "amount_sum":
            out = sum_q
        elif name == "vwap":
            # out = np.full(T, np.nan, dtype=np.float32)
            mask = sum_q > 0
            out[mask] = sum_pq[mask] / sum_q[mask]
        else:
            raise NotImplementedError(f"{name} not supported yet")
        out = out[k:]  # 只保留有效部分

        # step 3: empty_policy
        if empty_policy == "zero":
            out = np.nan_to_num(out, nan=0.0)
        elif empty_policy == "ffill":
            for i in range(1, T - k):
                if np.isnan(out[i]):
                    out[i] = out[i - 1]

        return torch.tensor(out[:, None], dtype=torch.float32, device=self.device)


    def aggregate(self, name: AggName, lookback_ms: int, empty_policy: EmptyPolicy = "nan") -> torch.Tensor:
        """
        主函数：根据 name 和 lookback_ms 计算聚合特征，返回 shape: [T,1] 的 tensor
        """
        self._check_lookback(lookback_ms)
        tensor = self._compute_tensor(name, lookback_ms, empty_policy)  # 返回 [T,1]

        return tensor