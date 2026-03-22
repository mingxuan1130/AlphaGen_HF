from __future__ import annotations

from typing import List, Literal
from enum import IntEnum
import os
import pandas as pd
import torch

from data.trades_aggregator import TradesAggregator, AggName, EmptyPolicy

from dataclasses import dataclass

class FeatureType(IntEnum):
    """
    引入到 features 中，告诉系统特征对应的 column index
    """
    ASK_1 = 0; ASK_2 = 1; ASK_3 = 2; ASK_4 = 3; ASK_5 = 4
    BID_1 = 5; BID_2 = 6; BID_3 = 7; BID_4 = 8; BID_5 = 9

AggName = Literal["amount_sum", "vwap"]
EmptyPolicy = Literal["nan", "zero", "ffill"]

@dataclass(frozen=True)
class HFConfig:
    instrument: str
    start_time: str
    end_time: str
    root_type: str = "data_share"
    subsample: str = "100ms"
    max_backtrack_ms: int = 1200
    max_future_ms: int = 600
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise TypeError("HFConfig.instrument must be a non-empty string for single-asset mode")

class HFDataManager:
    def __init__(self, cfg: HFConfig):
        self.cfg = cfg
        self.device = cfg.device
        self._base_dir = self._resolve_base_dir(cfg.root_type)  # 在这里统一解析
        self._snapshot = None
        self._trades = None
        self._trades_agg = None
        self._agg_cache = {}

    @staticmethod
    def _resolve_base_dir(root_type: str) -> str:
        if root_type == "data_share":
            return "data_share"
        elif root_type in {"local", "data_local"}:
            return "data_local"
        else:
            raise ValueError(
                f"Unknown root_type: {root_type}. Expected one of: data_share, local, data_local"
            )

    
    def _get_trades_agg(self):
        if self._trades_agg is None:
            s = self.snapshot
            t = self.trades.data
            self._trades_agg = TradesAggregator(
                snapshot_ts=s.snapshot_ts,
                trades_ts=t["ts"],
                trades_price=t["price"].to_numpy(),
                trades_amount=t["amount"].to_numpy(),
                snapshot_step_ms=s.snapshot_step_ms,
                device=self.device,
            )
        return self._trades_agg

    def trade_feature(self, name: AggName, lookback_ms: int,
                      period: slice = slice(0, 1),
                      empty_policy: EmptyPolicy = "nan"):
        start = period.start + self.snapshot.max_backtrack_ms
        stop = period.stop + self.snapshot.max_backtrack_ms + self.snapshot.n_ms - 1
        full = self._get_trades_agg().aggregate(name, lookback_ms, empty_policy=empty_policy)
        return full[start:stop]

    @property
    def snapshot(self):
        if self._snapshot is None:
            self._snapshot = SnapshotData(self.cfg, self._base_dir)  # 传递 base_dir
        return self._snapshot

    @property
    def snapshot_tensor(self) -> torch.Tensor:
        return self.snapshot.data  
    
    @property
    def snapshot_ts_start(self):
        return self.snapshot.snapshot_ts.iloc[0]

    @property
    def trades(self):
        if self._trades is None:
            self._trades = TradesData(self.cfg, self._base_dir)  # 传递 base_dir
        return self._trades
    
    @property
    def n_instruments(self) -> int:
        return 1

    @property
    def max_backtrack_ms(self) -> int:
        return self.cfg.max_backtrack_ms
    
    @property
    def max_future_ms(self) -> int:
        return self.cfg.max_future_ms
    
    @property
    def n_ms(self) -> int:
        return self.snapshot.n_ms
    
    


class SnapshotData:
    DEFAULT_FEATURES = [
        "asks[0].price", "asks[1].price", "asks[2].price", "asks[3].price", "asks[4].price",
        "bids[0].price", "bids[1].price", "bids[2].price", "bids[3].price", "bids[4].price",
    ]

    def __init__(self, cfg: HFConfig, base_dir: str, features: List[str] | None = None):
        self.cfg = cfg
        self.start_time = cfg.start_time
        self.end_time = cfg.end_time
        self.features = features or self.DEFAULT_FEATURES
        self.max_backtrack_ms = cfg.max_backtrack_ms
        self.max_future_ms = cfg.max_future_ms
        self.subsample = cfg.subsample
        self.device = cfg.device
        self._base_dir = base_dir  # 直接接收 base_dir

        self.data, _, _, self.data_df = self._load_data()
        self.snapshot_step_ms = int(pd.Timedelta(self.subsample).total_seconds() * 1000)

        self.n_ms = self.data.shape[0] - self.max_backtrack_ms - self.max_future_ms
        if self.n_ms <= 0:
            raise ValueError(f"snapshot rows不足以覆盖backtrack/future: data.shape[0]={self.data.shape[0]} rows, "
                             f"max_backtrack_ms={self.max_backtrack_ms}, max_future_ms={self.max_future_ms}")

    @property
    def n_features(self) -> int:
        return len(self.features)
    
    @property
    def snapshot_ts(self) -> pd.Series:
        return self.data_df["ts"]

    def _get_data_path(self) -> List[str]:
        start_dt = pd.to_datetime(self.start_time).normalize()
        end_dt = pd.to_datetime(self.end_time).normalize()

        if end_dt < start_dt:
            raise ValueError("end_time must be >= start_time")

        base_dir = self._base_dir
        all_paths: List[str] = []
        inst = self.cfg.instrument
        subdir = os.path.join(base_dir, inst, "book_snapshot_25")
        for d in pd.date_range(start_dt, end_dt, freq="D"):
            date_str = d.strftime("%Y-%m-%d")
            filename = f"binance-futures_book_snapshot_25_{date_str}_{inst}.csv.gz"
            path = os.path.join(subdir, filename)
            if os.path.exists(path):
                all_paths.append(path)

        return all_paths
    

    def _load_and_subsample(self, path: str) -> pd.DataFrame:
        """
        注意: 在合并过程中timestamp要一致
              trades和snapshot都有两个时间戳, timestamp和local_ts, 我们统一用timestamp
        """
        df = pd.read_csv(path)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["timestamp"], unit="us")
        df = df.resample(self.subsample, on="ts").last().reset_index().ffill()
        return df
    
    
    def _load_data(self) -> pd.DataFrame:
        """
        读取数据文件，返回 Tensor 格式的数据，以及对应的时间和币种索引
        """
        
        paths = self._get_data_path()
        if not paths:
            raise FileNotFoundError(f"No data files found in {paths}")

        dfs = []
        for p in paths:
            dfs.append(self._load_and_subsample(p))

        data = pd.concat(dfs, ignore_index=True)

        #用对应时间筛选币种数据
        if "ts" not in data.columns:
            raise KeyError(f"Missing 'ts' column in {data.columns.tolist()}")
        
        data["ts"] = pd.to_datetime(data["ts"], unit="us")
        data = data.sort_values("ts")

        start_ts = pd.to_datetime(self.start_time)
        end_ts = pd.to_datetime(self.end_time)
        data = data[(data["ts"] >= start_ts) & (data["ts"] <= end_ts)]

        missing = [c for c in self.features if c not in data.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing} in {data.columns.tolist()}")

        feature_values = data[self.features].values.astype("float32")
        feature_values = feature_values.reshape(feature_values.shape[0], feature_values.shape[1], 1)
        feature_tensor = torch.tensor(feature_values, dtype=torch.float, device=self.device)

        dates = pd.Index(data["ts"].astype(str).tolist())
        crypto_ids = pd.Index([self.cfg.instrument])

        return feature_tensor, dates, crypto_ids, data
    
class TradesData:
    def __init__(self, cfg: HFConfig, base_dir: str):
        self.start_time = cfg.start_time
        self.end_time = cfg.end_time
        self.cfg = cfg
        self.device = cfg.device
        self._base_dir = base_dir  # 直接接收 base_dir
        self.data = self._load_data()

    def _get_data_path(self) -> List[str]:
        start_dt = pd.to_datetime(self.start_time).normalize()
        end_dt = pd.to_datetime(self.end_time).normalize()

        if end_dt < start_dt:
            raise ValueError("end_time must be >= start_time")

        base_dir = self._base_dir
        all_paths: List[str] = []
        inst = self.cfg.instrument
        subdir = os.path.join(base_dir, inst, "trades")
        for d in pd.date_range(start_dt, end_dt, freq="D"):
            date_str = d.strftime("%Y-%m-%d")
            filename = f"binance-futures_trades_{date_str}_{inst}.csv.gz"
            path = os.path.join(subdir, filename)
            if os.path.exists(path):
                all_paths.append(path)

        return all_paths

    def _load_data(self):
        paths = self._get_data_path()
        if not paths:
            return pd.DataFrame()

        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            if df.empty:
                continue
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        
        # 添加时间筛选逻辑
        if "timestamp" not in data.columns:
            raise KeyError(f"Missing 'timestamp' column in trades data")
        
        # 转换时间戳并筛选
        data["ts"] = pd.to_datetime(data["timestamp"], unit="us")
        start_ts = pd.to_datetime(self.start_time)
        end_ts = pd.to_datetime(self.end_time)
        data = data[(data["ts"] >= start_ts) & (data["ts"] <= end_ts)]
        
        # # 转换回微秒整数（保持与 SnapshotData 一致）
        # data["timestamp"] = (data["timestamp"].astype(np.int64) // 1000).astype(np.int64)
        
        return data

    
    def validate_for_agg(self) -> None:
        required = {"ts", "price", "amount"}
        if self.data.empty:
            raise ValueError("Trades data is empty")
        if not required.issubset(self.data.columns):
            raise KeyError(f"Trades columns missing: {required - set(self.data.columns)}")

    @property
    def trades_ts(self):
        return self.data["ts"]

    @property
    def trades_price(self):
        return self.data["price"].to_numpy(np.float64)

    @property
    def trades_amount(self):
        return self.data["amount"].to_numpy(np.float64)
