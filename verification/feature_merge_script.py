import torch
from data.alphaHF_data import HFConfig, HFDataManager, FeatureType
from data.snapshot_expression import Feature, TradeVWAP

def main():
    # 跟你当前 snapshot_script 一致的配置
    INSTRUMENTS = "BTCUSDT"
    ROOT = "data_share"
    TRAIN_START = "2025-01-01"
    TRAIN_END = "2025-01-02"

    cfg = HFConfig(
        instrument=INSTRUMENTS,
        start_time=TRAIN_START,
        end_time=TRAIN_END,
        root_type=ROOT,
        subsample="100ms",
        device=torch.device("cpu"),
    )
    dm = HFDataManager(cfg)

    # snapshot 因子
    mid_px = (Feature(FeatureType.ASK_1) + Feature(FeatureType.BID_1)) / 2

    # trades 因子（例如 1000ms 的 VWAP）
    trade_vwap_1s = TradeVWAP(1000)

    # 组合：用 trades 构造的因子 + snapshot 因子
    expr = (trade_vwap_1s / mid_px) - 1

    # 只评估一小段，避免过大
    out = expr.evaluate(dm, period=slice(0, 10))

    print("expr output shape:", out.shape)
    print("expr output (first 5 rows):")
    print(out[:5])

if __name__ == "__main__":
    main()
