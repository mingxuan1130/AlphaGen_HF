from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Sequence
from torch import Tensor
import torch


from data.alphaHF_data import HFDataManager
from data.snapshot_expression import SnapshotExpression
from utils import series_pearsonr, series_spearmanr


class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: SnapshotExpression) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_mutual_IC(self, expr1: SnapshotExpression, expr2: SnapshotExpression) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: Sequence[SnapshotExpression], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: Sequence[SnapshotExpression], weights: Sequence[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'


class CryptoAlphaCalculator(AlphaCalculator):
    def __init__(self, dm: HFDataManager, target: Optional[SnapshotExpression] = None):
        if target is not None:
            raw_target = target.evaluate(dm)
            self.target = raw_target
        else:
            self.target = None
        self.dm = dm

    def _calc_alpha(self, expr: SnapshotExpression) -> Tensor:
        return expr.evaluate(self.dm)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        """
        清理NaN之后计算时序ICSem
        """
        v1 = value1.reshape(-1)
        v2 = value2.reshape(-1)
        valid_mask = (~torch.isnan(v1)) & (~torch.isnan(v2))
        if valid_mask.sum() >= 2:
            print("valid=", valid_mask.sum().item(), "len=", v1.numel(),
                  "v1_std=", v1.std().item(), "v2_std=", v2.std().item())
        return series_pearsonr(value1, value2).item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        """
        清理NaN之后计算时序 Rank IC
        """
        return series_spearmanr(value1, value2).item()

    
    def make_ensemble_alpha(self, exprs: Sequence[SnapshotExpression], weights: Sequence[float]) -> Tensor:
        n = len(exprs)
        factors = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return torch.sum(torch.stack(factors, dim=0), dim=0)
    
    def calc_single_IC_ret(self, expr: SnapshotExpression) -> float:
        return self._calc_IC(self._calc_alpha(expr), self.target)
    
    def calc_mutual_IC(self, expr1: SnapshotExpression, expr2: SnapshotExpression) -> float:
        return self._calc_IC(self._calc_alpha(expr1), self._calc_alpha(expr2))
    
    def calc_pool_IC_ret(self, exprs: Sequence[SnapshotExpression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(value, self.target)
        
    def calc_pool_rIC_ret(self, exprs: Sequence[SnapshotExpression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(value, self.target)

    
    def calc_single_IC_ret_daily(self, expr: SnapshotExpression) -> Tensor:
        return series_pearsonr(self._calc_alpha(expr), self.target)
    
    def calc_single_rIC_ret(self, expr: SnapshotExpression) -> float:
        return self._calc_rIC(self._calc_alpha(expr), self.target)
    
    def calc_single_all_ret(self, expr: SnapshotExpression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        target = self.target
        return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_mutual_IC_daily(self, expr1: SnapshotExpression, expr2: SnapshotExpression) -> Tensor:
        return series_pearsonr(self._calc_alpha(expr1), self._calc_alpha(expr2))
