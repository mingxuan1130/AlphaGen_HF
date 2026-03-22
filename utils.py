import torch
from torch import Tensor


def _rank_data_1d(x: Tensor) -> Tensor:
    _, inv, counts = x.unique(return_inverse=True, return_counts=True)
    cs = counts.cumsum(dim=0)
    cs = torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), cs))
    rmin = cs[:-1]
    rmax = cs[1:] - 1
    ranks = (rmin + rmax) / 2
    return ranks[inv]


def series_pearsonr(x: Tensor, y: Tensor, min_count: int = 2) -> Tensor:
    """
    Time-series Pearson correlation for flattened vectors with NaN masking.
    Returns a scalar tensor.

    注意 pytorch 的 x.mean() 和 x.std() 的有偏和无偏默认是不一样的
    - x.mean() 是有偏的，除以 n
    - x.std() 是无偏的，除以 n-1
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    valid = (~x.isnan()) & (~y.isnan())
    if valid.sum() < min_count:
        return torch.tensor(0.0, device=x.device)
    x = x[valid]
    y = y[valid]
    x = x - x.mean()
    y = y - y.mean()
    # 用有偏标准差计算, 和 mean 一致
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom < 1e-12:
        return torch.tensor(0.0, device=x.device)
    return (x * y).mean() / denom


def series_spearmanr(x: Tensor, y: Tensor, min_count: int = 2) -> Tensor:
    """
    Time-series Spearman correlation for flattened vectors with NaN masking.
    Returns a scalar tensor.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    valid = (~x.isnan()) & (~y.isnan())
    if valid.sum() < min_count:
        return torch.tensor(0.0, device=x.device)
    x = x[valid]
    y = y[valid]
    rx = _rank_data_1d(x)
    ry = _rank_data_1d(y)
    return series_pearsonr(rx, ry, min_count=min_count)
