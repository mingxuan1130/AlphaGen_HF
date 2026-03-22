import pandas as pd

from verification.validation_data_generator import (
    generate_time_segments,
    load_data_pairs,
)
from data.snapshot_expression import TradeVWAP, TradeAmount
from data.alphaHF_data import HFDataManager, HFConfig

# 步骤1：生成时间段
time_segments = generate_time_segments(
    start_time='2025-01-05 00:00:05.300',
    durations=[1000, 2000],
    segments_per_duration=1, 
    gap_ms=0 # 必须设置为100ms的整数倍 
)

# 步骤2：加载数据
data_pairs = load_data_pairs(
    time_segments=time_segments,
    instrument="BTCUSDT",
    root_type="data_share"
)

def process_data_pair(data_pair, pair_index):
    """
    处理单个数据对，计算并对比 TradeAmount 和 TradeVWAP
    
    Args:
        data_pair: 包含 snapshot 和 trades 数据的字典
        pair_index: 数据对的索引（用于显示）
    """
    print(f"\n{'='*60}")
    print(f"处理数据对 [{pair_index}]: {data_pair['duration_ms']}ms #{data_pair['segment_id']}")
    print(f"时间范围: {data_pair['start_time']} to {data_pair['end_time']}")
    print(f"{'='*60}")

    # ========= 计算 expr 版本的 TradeAmount =========
    cfg = HFConfig(
        instrument="BTCUSDT",
        start_time=data_pair['start_time'],
        end_time=data_pair['end_time'],
        root_type="data_share",
        subsample="100ms",
        max_backtrack_ms=0,
        max_future_ms=0
    )

    dm = HFDataManager(cfg)

    # 验证对比时禁用 factor cache，避免命中历史结果导致错判
    # snapshot_data.cache_dir = None
    # snapshot_data._trades_agg = None
    
    # 计算 TradeAmount 表达式结果
    ta_100ms_expr = TradeAmount(100).evaluate(dm, period=slice(0, 1))
    ta_200ms_expr = TradeAmount(200).evaluate(dm, period=slice(0, 1))
    ta_500ms_expr = TradeAmount(500).evaluate(dm, period=slice(0, 1))
    
    ta_100ms_expr = ta_100ms_expr.squeeze(-1).numpy()
    ta_200ms_expr = ta_200ms_expr.squeeze(-1).numpy()
    ta_500ms_expr = ta_500ms_expr.squeeze(-1).numpy()
    
    # ========= 计算 Pandas 版本的 TradeAmount =========
    # 1. 拿到 trades 数据的 DataFrame，并设置时间戳为索引
    # 2. 使用 resample 进行时间重采样，计算每个时间段内的成交量总和
    # 注意：resample 的开始时间必须是 snapshot 的开始时间
    # ==================================================

    df = dm.trades.data
    df_set_idx = df.set_index('ts', inplace=False)

    # 注意：这里时间戳必须是snapshot的时间戳，因为trades的时间戳是没对齐的
    start_time = dm.snapshot_ts_start
    
    # origin=start_time 用于将trades数据对齐到snapshot上
    sum_amount_100 = df_set_idx['amount'].resample(
        '100ms', origin=start_time, closed='right', label='right'
    ).sum()
    
    sum_amount_200 = pd.Series(sum_amount_100).rolling(window=2).sum().dropna().values
    sum_amount_500 = pd.Series(sum_amount_100).rolling(window=5).sum().dropna().values
    
    print("\n[TradeAmount] 对比结果：")
    print("TradeAmount 100ms:")
    print("  表达式:", ta_100ms_expr)
    print("  Pandas: ", sum_amount_100.values)
    print("TradeAmount 200ms:")
    print("  表达式:", ta_200ms_expr)
    print("  Pandas: ", sum_amount_200)
    print("TradeAmount 500ms:")
    print("  表达式:", ta_500ms_expr)
    print("  Pandas: ", sum_amount_500)
    
    # 计算 TradeVWAP 表达式结果
    vwap_100ms_expr = TradeVWAP(100).evaluate(dm, period=slice(0, 1))
    vwap_200ms_expr = TradeVWAP(200).evaluate(dm, period=slice(0, 1))
    vwap_500ms_expr = TradeVWAP(500).evaluate(dm, period=slice(0, 1))
    
    # vwap_100ms_expr = vwap_100ms_expr.squeeze(-1).numpy()[1:]
    # vwap_200ms_expr = vwap_200ms_expr.squeeze(-1).numpy()[2:]
    # vwap_500ms_expr = vwap_500ms_expr.squeeze(-1).numpy()[5:]
    vwap_100ms_expr = vwap_100ms_expr.squeeze(-1).numpy()
    vwap_200ms_expr = vwap_200ms_expr.squeeze(-1).numpy()
    vwap_500ms_expr = vwap_500ms_expr.squeeze(-1).numpy()
    
    # 计算 VWAP = Σ(price * amount) / Σ(amount)
    # 先计算每笔交易的成交额（价格 * 数量）
    df_set_idx['turnover'] = df_set_idx['price'] * df_set_idx['amount']
    
    # 对成交额进行resample
    sum_turnover_100 = df_set_idx['turnover'].resample(
        '100ms', origin=start_time, closed='right', label='right'
    ).sum()
    
    # 计算滚动VWAP
    sum_turnover_200 = pd.Series(sum_turnover_100).rolling(window=2).sum().dropna()
    sum_turnover_500 = pd.Series(sum_turnover_100).rolling(window=5).sum().dropna()
    
    # VWAP = 成交额 / 成交量（保留两位小数）
    vwap_100ms = (sum_turnover_100 / sum_amount_100).round(2).values
    vwap_200ms = (sum_turnover_200 / sum_amount_200).round(2).values
    vwap_500ms = (sum_turnover_500 / sum_amount_500).round(2).values
    
    print("\n[TradeVWAP] 对比结果：")
    print("TradeVWAP 100ms:")
    print("  表达式:", vwap_100ms_expr)
    print("  Pandas: ", vwap_100ms)
    print("TradeVWAP 200ms:")
    print("  表达式:", vwap_200ms_expr)
    print("  Pandas: ", vwap_200ms)
    print("TradeVWAP 500ms:")
    print("  表达式:", vwap_500ms_expr)
    print("  Pandas: ", vwap_500ms)


# 主流程
if __name__ == "__main__":
    print("\n数据对列表：")
    for i, pair in enumerate(data_pairs):
        print(f"{i}: {pair['duration_ms']}ms #{pair['segment_id']} - "
                f"{pair['start_time']} to {pair['end_time']}")

    # 循环处理所有数据对
    for i, data_pair in enumerate(data_pairs):
        process_data_pair(data_pair, i)
