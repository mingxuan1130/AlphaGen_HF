from abc import ABCMeta, abstractmethod
from typing import List, Type, Union

import torch
from torch import Tensor

from data.alphaHF_data import FeatureType, HFDataManager

class OutOfDataRangeError(IndexError):
    pass

class SnapshotExpression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: Union["SnapshotExpression", float]) -> "Add":
        if isinstance(other, SnapshotExpression):
            return Add(self, other)
        else:
            return Add(self, Constant(other))

    def __radd__(self, other: float) -> "Add": return Add(Constant(other), self)

    def __sub__(self, other: Union["SnapshotExpression", float]) -> "Sub":
        if isinstance(other, SnapshotExpression):
            return Sub(self, other)
        else:
            return Sub(self, Constant(other))

    def __rsub__(self, other: float) -> "Sub": return Sub(Constant(other), self)

    def __mul__(self, other: Union["SnapshotExpression", float]) -> "Mul":
        if isinstance(other, SnapshotExpression):
            return Mul(self, other)
        else:
            return Mul(self, Constant(other))

    def __rmul__(self, other: float) -> "Mul": return Mul(Constant(other), self)

    def __truediv__(self, other: Union["SnapshotExpression", float]) -> "Div":
        if isinstance(other, SnapshotExpression):
            return Div(self, other)
        else:
            return Div(self, Constant(other))

    def __rtruediv__(self, other: float) -> "Div": return Div(Constant(other), self)

    def __pow__(self, other: Union["SnapshotExpression", float]) -> "Pow":
        if isinstance(other, SnapshotExpression):
            return Pow(self, other)
        else:
            return Pow(self, Constant(other))

    def __rpow__(self, other: float) -> "Pow": return Pow(Constant(other), self)

    def __pos__(self) -> "SnapshotExpression": return self
    def __neg__(self) -> "Sub": return Sub(Constant(0), self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    def is_featured(self): raise NotImplementedError

# =============================================================================================
# ========================================  抽象类定义 ==========================================
# =============================================================================================

class Feature(SnapshotExpression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, dm: HFDataManager, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -dm.max_backtrack_ms or
                period.stop - 1 > dm.max_future_ms):
            raise OutOfDataRangeError()
        start = period.start + dm.max_backtrack_ms
        stop = period.stop + dm.max_backtrack_ms + dm.n_ms - 1
        return dm.snapshot.data[start:stop, int(self._feature), :]

    def __str__(self) -> str: return '$' + self._feature.name.lower()

    @property
    def is_featured(self): return True

class Constant(SnapshotExpression):
    def __init__(self, value: float) -> None:
        self._value = value

    def evaluate(self, dm: HFDataManager, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -dm.max_backtrack_ms or
                period.stop - 1 > dm.max_future_ms):
            raise OutOfDataRangeError()
        device = dm.device
        dtype = dm.snapshot.data.dtype
        ms = period.stop - period.start - 1 + dm.n_ms
        return torch.full(size=(ms, dm.n_instruments),
                          fill_value=self._value, dtype=dtype, device=device)

    def __str__(self) -> str: return f'Constant({str(self._value)})'

    @property
    def is_featured(self): return False

class TimeArg(SnapshotExpression):
    def evaluate(self, dm: HFDataManager, period=slice(0, 1)):
        raise AssertionError("TimeArg should not be evaluated")

    @property
    def is_featured(self):
        return False

class DeltaTime(TimeArg):
    def __init__(self, delta_time: int):
        self._delta_time = delta_time

class LookbackTime(TimeArg):
    def __init__(self, lookback_time: int):
        self._lookback_time = lookback_time

# Operator base classes
class Operator(SnapshotExpression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...

class TradeWindowOperator(Operator):
    def __init__(self, lookback_ms: Union[int, LookbackTime]) -> None:
        if isinstance(lookback_ms, LookbackTime):
            lookback_ms = lookback_ms._lookback_time
        self._lookback_ms = int(lookback_ms)

    @classmethod
    def n_args(cls) -> int:
        return 1  # 只吃一个 LookbackTime token

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return TradeWindowOperator

    @property
    def is_featured(self):
        return True

    @abstractmethod
    def _agg_name(self) -> str: ...

    def evaluate(self, dm: HFDataManager, period: slice = slice(0, 1)):
        return dm.trade_feature(
            self._agg_name(),
            self._lookback_ms,
            period=period,
            empty_policy="nan",
        )

    def __str__(self):
        return f"{type(self).__name__}({self._lookback_ms})"

# Operator implementations
class TradeVWAP(TradeWindowOperator):
    def _agg_name(self) -> str:
        return "vwap"

class TradeHigh(TradeWindowOperator):
    def _agg_name(self) -> str:
        raise NotImplementedError("TradeHigh is not implemented yet")
        # return "price_high"

class TradeLow(TradeWindowOperator):
    def _agg_name(self) -> str:
        raise NotImplementedError("TradeLow is not implemented yet")
        # return "price_low"

class TradeAmount(TradeWindowOperator):
    def _agg_name(self) -> str:
        return "amount_sum"
    
Trades_classes: List[Type[SnapshotExpression]] = [TradeVWAP, TradeAmount]

class UnaryOperator(Operator):
    def __init__(self, operand: Union[SnapshotExpression, float]) -> None:
        self._operand = operand if isinstance(operand, SnapshotExpression) else Constant(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: return UnaryOperator

    def evaluate(self, dm: HFDataManager, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(dm, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_featured(self): return self._operand.is_featured

class BinaryOperator(Operator):
    def __init__(self, lhs: Union[SnapshotExpression, float], rhs: Union[SnapshotExpression, float]) -> None:
        self._lhs = lhs if isinstance(lhs, SnapshotExpression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, SnapshotExpression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return BinaryOperator

    def evaluate(self, dm: HFDataManager, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(dm, period), self._rhs.evaluate(dm, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured

class RollingOperator(Operator):
    def __init__(self, operand: Union[SnapshotExpression, float], delta_time: Union[int, DeltaTime]) -> None:
        self._operand = operand if isinstance(operand, SnapshotExpression) else Constant(operand)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingOperator

    def evaluate(self, dm: HFDataManager, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = self._operand.evaluate(dm, slice(start, stop))   # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
        return self._apply(values)                                  # (L, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time})"

    @property
    def is_featured(self): return self._operand.is_featured

class PairRollingOperator(Operator):
    def __init__(self,
                 lhs: SnapshotExpression, rhs: SnapshotExpression,
                 delta_time: Union[int, DeltaTime]) -> None:
        self._lhs = lhs if isinstance(lhs, SnapshotExpression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, SnapshotExpression) else Constant(rhs)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return PairRollingOperator

    def _unfold_one(self, expr: SnapshotExpression,
                    dm: HFDataManager, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = expr.evaluate(dm, slice(start, stop))            # (L+W-1, S)
        return values.unfold(0, self._delta_time, 1)                # (L, S, W)

    def evaluate(self, dm: HFDataManager, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, dm, period)
        rhs = self._unfold_one(self._rhs, dm, period)
        return self._apply(lhs, rhs)                                # (L, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# =============================================================================================
# =======================================  操作符实现 ==========================================
# =============================================================================================

# =========================
# Unary Operators 
# =========================

# 原有部分
class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: 
        out = operand.log()
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        out[out.isnan()] = 0
        return out

class CSRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = 0
        return rank

# 新增部分
class Sqrt(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = operand.sqrt()
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Power2(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand ** 2

class Power3(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand ** 3

class Inv(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = 1 / operand
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Neg(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return -operand

class Sigmoid(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = torch.sigmoid(operand)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out
        
class Tanh(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out =  torch.tanh(operand)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Softmax(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = torch.softmax(operand, dim=-1)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Relu(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = torch.relu(operand)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Leaky_relu(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = torch.nn.functional.leaky_relu(operand)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Elu(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = torch.nn.functional.elu(operand)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class S_log_1p(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 对输入张量应用 log(1 + x) 操作
        out = torch.log1p(operand)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Arc_tanh(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = torch.atanh(operand)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Signed_power(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = torch.sign(operand) * operand ** 2
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out
    
# class Half(UnaryOperator):
#     def _apply(self, operand: Tensor) -> Tensor: return operand.exp()
Unary_operator_classes: List[Type[SnapshotExpression]] = [
    Abs, Sign, Log, CSRank, Sqrt, Power2, Power3, Inv, Neg, 
    Sigmoid, Tanh, Softmax, Relu, Leaky_relu, Elu, S_log_1p, Arc_tanh, 
    Signed_power
]

# =========================
# Binary Operators 
# =========================
# 原有部分
class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs

class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs

class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs

class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: 
        out = lhs / rhs
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        out[out.isnan()] = 0
        return out

class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: 
        out = lhs ** rhs
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        out[out.isnan()] = 0
        
        return out

class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured

class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured
    
# 新增部分
class Max2(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.max(lhs, rhs)

class Min2(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.min(lhs, rhs)

class Ortho(BinaryOperator):
    """
    判断两个向量是否正交
    """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        elementwise_product = lhs * rhs
        inner_product = torch.sum(elementwise_product, dim=-1, keepdim=True)
        orthogonal = torch.isclose(inner_product, torch.zeros_like(inner_product))
        orthogonal = orthogonal.expand_as(lhs)
        orthogonal = orthogonal.float()
        return orthogonal

Binary_operator_classes: List[Type[SnapshotExpression]] = [
    Add, Sub, Mul, Div, Pow, Greater, Less, Max2, Min2, Ortho
]

# =========================
# Rolling Operators 
# =========================
# 原有部分
class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...

class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)

class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)

class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)

class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: 
        out = operand.var(dim=-1)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5

class Kurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        out = m4 / var ** 2 - 3
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]

class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]

class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]  #因为.median返回两个Tensor，第一个是中位数，第二个是中位数的位置

class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)

class Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result

class Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...

class WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        out = (weights * operand).sum(dim=-1)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        out = (weights * operand).sum(dim=-1)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

# 新增部分
class Curt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 计算中心化数据
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1, unbiased=False)
        kurt = m4 / var ** 2 - 3
        out = kurt
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out
    
class Demean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[:, :, -1] - operand.mean(dim=-1)

class Ts_zscore(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        mean = operand.mean(dim=-1)
        std = operand.std(dim=-1)
        return (operand[:, :, -1] - mean) / std
    
class Ts_rank_gmean(RollingOperator):
    """
    几何平均时序排名
    """
    def _apply(self, operand: Tensor) -> Tensor:
        rank = operand.argsort().argsort()
        rank = rank.float()
        rank = rank / rank.shape[-1]
        out = torch.sqrt(rank.mean(dim=-1))
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_rank_amean(RollingOperator):
    """
    算术平均时序排名
    """
    def _apply(self, operand: Tensor) -> Tensor:
        rank = operand.argsort().argsort()
        rank = rank.float()
        rank = rank / rank.shape[-1]
        return rank.mean(dim=-1)

class Ts_rank_gmean_amean_diff(RollingOperator):
    """
    几何平均时序排名与算术平均时序排名之差
    """
    def _apply(self, operand: Tensor) -> Tensor:
        rank = operand.argsort().argsort()
        rank = rank.float()
        rank = rank / rank.shape[-1]
        out = torch.sqrt(rank.mean(dim=-1)) - rank.mean(dim=-1)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_ir(RollingOperator):
    """
    ir(x) = mean(x) / std(x)
    """
    def _apply(self, operand: Tensor) -> Tensor:
        excess_return_mean = operand.mean(dim=-1)
        excess_return_std = operand.std(dim=-1)
        ir = excess_return_mean / excess_return_std
        ir[excess_return_std == 0] = 0
        return ir

class Ts_monoton(RollingOperator):
    """
    单调程度评价
    """
    def _apply(self, operand: Tensor) -> Tensor:
        # 计算每个窗口的变化率
        change_rate = operand[:, :, 0] / operand[:, :, -1] - 1
        change_rate[change_rate.isnan()] = 0
        change_rate[change_rate == float('inf')] = 0
        change_rate[change_rate == float('-inf')] = 0
        return change_rate

class Ts_max_to_min(RollingOperator):
    """
    max_to_min(x) = max(x) / min(x)
    """
    def _apply(self, operand: Tensor) -> Tensor:
        max_value = operand.max(dim=-1)[0]
        min_value = operand.min(dim=-1)[0]
        out = max_value / min_value
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_maxmin_norm(RollingOperator):
    """
    maxmin_norm(x) = (x[-1] - min(x)) / (max(x) - min(x))
    """
    def _apply(self, operand: Tensor) -> Tensor:
        max_value = operand.max(dim=-1)[0]
        min_value = operand.min(dim=-1)[0]
        out = (operand[:, :, -1] - min_value) / (max_value - min_value)
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out
    
class Ts_last_to_max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        max_value = operand.max(dim=-1)[0]
        out = operand[:, :, -1] / max_value
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_last_to_min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        min_value = operand.min(dim=-1)[0]
        out = operand[:, :, -1] / min_value
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out
    
class Ts_last_to_mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        mean_value = operand.mean(dim=-1)
        out = operand[:, :, -1] / mean_value
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_last_to_ewm(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        ema = (weights * operand).sum(dim=-1)
        out = operand[:, :, -1] / ema
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_last_to_wmean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        wmean = (weights * operand).sum(dim=-1)
        out = operand[:, :, -1] / wmean
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_min_max_cps(RollingOperator):
    """
    最大值的位置和最小值的位置交替出现的次数占总长度的比例
    """
    def _apply(self, operand: Tensor) -> Tensor:
        # 确定最小值和最大值的位置
        min_positions = operand == operand.min(dim=-1, keepdim=True).values
        max_positions = operand == operand.max(dim=-1, keepdim=True).values
        # 计算交叉点（即最小值和最大值交替出现的位置）
        crossover_points = min_positions ^ max_positions
        crossover_count = crossover_points.sum(dim=-1)
        cps_ratio = crossover_count.float() / operand.shape[-1]
        return cps_ratio

class Ts_min_max_diff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        max_value = operand.max(dim=-1)[0]
        min_value = operand.min(dim=-1)[0]
        return max_value - min_value

class Ts_pctchg_abs(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = operand[:, :, -1] / operand[:, :, 0] - 1
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out.abs()

class Ts_pctchg(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = operand[:, :, -1] / operand[:, :, 0] - 1
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_log_pctchg(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        operand_adj = (operand + 1).log()
        out = operand_adj[:, :, -1] / operand_adj[:, :, 0] - 1
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Ts_hhi(RollingOperator):
    """
    集中度指数，看给定窗口内的值是否集中在某几个点上
    """
    def _apply(self, operand: Tensor) -> Tensor:
        # 将每个窗口中的值归一化（使它们的总和为1）
        normalized = operand / operand.sum(dim=-1, keepdim=True)
        squared = normalized ** 2
        hhi = squared.sum(dim=-1)
        return hhi

class Ts_rsi(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        delta = operand[:, :, 1:] - operand[:, :, :-1]
        gains = torch.where(delta > 0, delta, torch.tensor(0.0))
        losses = torch.where(delta < 0, -delta, torch.tensor(0.0))
        avg_gain = gains.mean(dim=-1)
        avg_loss = losses.mean(dim=-1)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100  # 如果没有下跌，RSI设为100
        return rsi / 100

class Ts_winsorize(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 设置Winsorization的上下界百分位数
        lower_percentile = 0.05
        upper_percentile = 0.95
        lower_bound = torch.quantile(operand, lower_percentile, dim=-1, keepdim=True)
        upper_bound = torch.quantile(operand, upper_percentile, dim=-1, keepdim=True)
        winsorized = torch.where(operand < lower_bound, lower_bound, operand)
        winsorized = torch.where(operand > upper_bound, upper_bound, winsorized)

        return winsorized[:, :, -1]

class Ts_quantile(RollingOperator):
    #TODO: 这个操作需要拓展
    def _apply(self, operand: Tensor) -> Tensor:
        # 设置分位数
        percentile = 0.75
        lower_bound = torch.quantile(operand, percentile, dim=-1, keepdim=True)
        return lower_bound[:, :, -1]

class Ts_rank_by_side(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        positive_mask = operand > 0
        negative_mask = operand < 0
        positive_rank = positive_mask.float() * operand.argsort(dim=-1).float()
        negative_rank = negative_mask.float() * operand.argsort(dim=-1, descending=True).float()
        combined_rank = positive_rank + negative_rank

        return combined_rank[:, :, -1]

class Ts_fraction(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        out = operand[:, :, 0] / operand[:, :, -1]
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out
    
Rolling_operator_classes: List[Type[SnapshotExpression]] = [
    Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min, Med, Mad, Rank, Delta, WMA, EMA, Curt, Demean,
    Ts_zscore, Ts_rank_gmean, Ts_rank_amean, Ts_rank_gmean_amean_diff, Ts_ir, Ts_monoton, Ts_max_to_min, 
    Ts_maxmin_norm, Ts_last_to_max, Ts_last_to_min, Ts_last_to_mean, Ts_last_to_ewm, Ts_last_to_wmean, 
    Ts_min_max_cps, Ts_min_max_diff, Ts_pctchg_abs, Ts_pctchg, Ts_log_pctchg, Ts_hhi, Ts_rsi, Ts_winsorize, 
    Ts_quantile, Ts_rank_by_side, Ts_fraction
]   
# =========================
# PairRolling Operators 
# =========================
# 原有部分
class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul
    
# 新增部分
class Ts_rankcorr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        clhs_rank = lhs_rank - lhs_rank.mean(dim=-1, keepdim=True)
        crhs_rank = rhs_rank - rhs_rank.mean(dim=-1, keepdim=True)
        rankcov = (clhs_rank * crhs_rank).sum(dim=-1)
        lhs_std = clhs_rank.std(dim=-1)
        rhs_std = crhs_rank.std(dim=-1)
        rankcorr = rankcov / (lhs_std * rhs_std)
        rankcorr[(lhs_std == 0) | (rhs_std == 0)] = 0

        return rankcorr

class Ts_cokurt(PairRollingOperator):
    """ 协同峰度 """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        n = lhs.shape[-1]
        numerator = ((clhs * crhs) ** 2).sum(dim=-1)
        denominator_lhs = (clhs ** 2).sum(dim=-1)
        denominator_rhs = (crhs ** 2).sum(dim=-1)
        cokurt = numerator / (denominator_lhs * denominator_rhs) * n**2 / ((n-1)*(n-2)*(n-3))
        cokurt[denominator_lhs * denominator_rhs == 0] = 0
        return cokurt

class Ts_coskew(PairRollingOperator):
    """ 协同偏度 """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        n = lhs.shape[-1]
        numerator = (clhs * clhs * crhs).sum(dim=-1)
        denominator_lhs = (clhs ** 2).sum(dim=-1).sqrt()
        denominator_rhs = (crhs ** 2).sum(dim=-1).sqrt()
        coskew = numerator / (denominator_lhs * denominator_rhs) * n / ((n-1)*(n-2))
        coskew[(denominator_lhs * denominator_rhs) == 0] = 0
        return coskew

class Ts_fxcut_75(PairRollingOperator):
    """ 75%百分位数的时序分位数剪切 """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_percentile_75 = torch.quantile(lhs, 0.75, dim=-1, keepdim=True)
        rhs_percentile_75 = torch.quantile(rhs, 0.75, dim=-1, keepdim=True)
        comparison = lhs_percentile_75 / rhs_percentile_75
        comparison[comparison.isnan()] = 0
        comparison[comparison == float('inf')] = 0
        comparison[comparison == float('-inf')] = 0
        # drop the third dim
        comparison = comparison.squeeze(-1)
        return comparison

class Ts_fxcut_50(PairRollingOperator):
    """ 50%百分位数的时序分位数剪切 """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_percentile_50 = torch.quantile(lhs, 0.5, dim=-1, keepdim=True)
        rhs_percentile_50 = torch.quantile(rhs, 0.5, dim=-1, keepdim=True)
        comparison = lhs_percentile_50 / rhs_percentile_50
        comparison[comparison.isnan()] = 0
        comparison[comparison == float('inf')] = 0
        comparison[comparison == float('-inf')] = 0
        # drop the third dim
        comparison = comparison.squeeze(-1)
        return comparison

class Ts_fxumr_75(PairRollingOperator):
    """ 75%百分位数的时序分位数差 """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # 计算左侧和右侧张量的75%百分位数
        lhs_percentile_75 = torch.quantile(lhs, 0.75, dim=-1, keepdim=True)
        rhs_percentile_75 = torch.quantile(rhs, 0.75, dim=-1, keepdim=True)
        result = lhs_percentile_75 - rhs_percentile_75
        result = result.squeeze(-1)
        return result

class Ts_fxumr_50(PairRollingOperator):
    """ 50%百分位数的时序分位数差 """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # 计算左侧和右侧张量的50%百分位数
        lhs_percentile_50 = torch.quantile(lhs, 0.5, dim=-1, keepdim=True)
        rhs_percentile_50 = torch.quantile(rhs, 0.5, dim=-1, keepdim=True)
        result = lhs_percentile_50 - rhs_percentile_50
        result = result.squeeze(-1)
        return result
    
class Rank_add(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        lhs_rank = lhs_rank / lhs_rank.shape[-1]
        rhs_rank = rhs_rank / rhs_rank.shape[-1]
        return (lhs_rank[:,:, -1] + rhs_rank[:,:, -1]) / lhs_rank.shape[-1]*2

class Rank_mul(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        lhs_rank = lhs_rank / lhs_rank.shape[-1]
        rhs_rank = rhs_rank / rhs_rank.shape[-1]
        return (lhs_rank[:,:, -1] * rhs_rank[:,:, -1])

class Rank_sub(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        lhs_rank = lhs_rank / lhs_rank.shape[-1]
        rhs_rank = rhs_rank / rhs_rank.shape[-1]
        return (lhs_rank[:,:, -1] - rhs_rank[:,:, -1])

class Rank_div(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        lhs_rank = lhs_rank / lhs_rank.shape[-1]
        rhs_rank = rhs_rank / rhs_rank.shape[-1]
        out = (lhs_rank[:,:, -1] / rhs_rank[:,:, -1])
        out[(lhs_rank[:,:, -1] == 0) | (rhs_rank[:,:, -1] == 0)] = torch.nan
        return out

class Rank_amean(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        lhs_rank = lhs_rank / lhs_rank.shape[-1]
        rhs_rank = rhs_rank / rhs_rank.shape[-1]
        return (lhs_rank[:,:, -1] + rhs_rank[:,:, -1]) / 2

class Rank_gmean(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        lhs_rank = lhs_rank / lhs_rank.shape[-1]
        rhs_rank = rhs_rank / rhs_rank.shape[-1]
        out = torch.sqrt(lhs_rank[:,:, -1] * rhs_rank[:,:, -1])
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

class Rank_gmean_amean_diff(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_rank = lhs.argsort().argsort()
        rhs_rank = rhs.argsort().argsort()
        lhs_rank = lhs_rank.float()
        rhs_rank = rhs_rank.float()
        lhs_rank = lhs_rank / lhs_rank.shape[-1]
        rhs_rank = rhs_rank / rhs_rank.shape[-1]
        out = torch.sqrt(lhs_rank[:,:, -1] * rhs_rank[:,:, -1]) - (lhs_rank[:,:, -1] + rhs_rank[:,:, -1]) / 2
        out[out.isnan()] = 0
        out[out == float('inf')] = 0
        out[out == float('-inf')] = 0
        return out

Pair_rolling_operator_classes: List[Type[SnapshotExpression]] = [
    Cov, Corr, Ts_rankcorr, Ts_cokurt, Ts_coskew, Ts_fxcut_75, Ts_fxcut_50, Ts_fxumr_75, Ts_fxumr_50, 
    Rank_add, Rank_mul, Rank_sub, Rank_div, Rank_amean, Rank_gmean, Rank_gmean_amean_diff
]

Operators = (
    Unary_operator_classes
    + Binary_operator_classes
    + Rolling_operator_classes
    + Pair_rolling_operator_classes
    + Trades_classes
)
