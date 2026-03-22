import pandas as pd

from data.alphaHF_data import HFConfig, HFDataManager

def generate_time_segments(start_time, durations, segments_per_duration=2, gap_ms=50):
    """
    生成时间段信息
    
    参数:
        start_time: str 或 pd.Timestamp, 起始时间 (例如: '2025-01-05 00:00:05.300')
        durations: list of int, 时间段长度列表，单位毫秒 (例如: [100, 200, 150])
        segments_per_duration: int, 每个时长生成的数据段数量 (默认: 2)
        gap_ms: int, 相邻时间段之间的间隔，单位毫秒 (默认: 50)
    
    返回:
        pd.DataFrame: 时间段信息，包含以下列：
            - duration_ms: int, 时间段长度
            - segment_id: int, 段序号
            - start_time: str, 起始时间
            - end_time: str, 结束时间
    """
    # 转换起始时间
    start_ts = pd.Timestamp(start_time)
    
    # 存储时间段
    time_segments = []
    current_offset = pd.Timedelta(0)
    
    for duration_ms in durations:
        duration = pd.Timedelta(milliseconds=duration_ms)
        
        for i in range(segments_per_duration):
            segment_start = start_ts + current_offset
            segment_end = segment_start + duration
            
            time_segments.append({
                'duration_ms': duration_ms,
                'segment_id': i + 1,
                'start_time': segment_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'end_time': segment_end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            })
            
            # 为下一个时间段增加偏移（避免重叠）
            current_offset += duration + pd.Timedelta(milliseconds=gap_ms)
    
    # 转换为 DataFrame
    time_segments_df = pd.DataFrame(time_segments)
    
    print(f"生成了 {len(time_segments_df)} 个时间段：")
    print(time_segments_df)
    
    return time_segments_df


def load_data_pairs(time_segments, instrument="BTCUSDT", root_type="data_share", 
                    max_backtrack_ms=0, max_future_ms=0):
    """
    根据时间段信息批量加载数据对
    """
    # 存储所有数据对
    data_pairs = []
    
    print(f"\n开始加载 {len(time_segments)} 对数据...\n")
    
    # 使用时间段信息读取数据
    for idx, segment in time_segments.iterrows():
        print(f"{'='*60}")
        print(f"[{idx+1}/{len(time_segments)}] 读取 {segment['duration_ms']}ms 时间段 #{segment['segment_id']}")
        print(f"起始时间: {segment['start_time']}")
        print(f"结束时间: {segment['end_time']}")
        print(f"{'='*60}")
        
        try:
            cfg = HFConfig(
                instrument=instrument,
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                root_type=root_type,
                max_backtrack_ms=max_backtrack_ms,
                max_future_ms=max_future_ms,
            )
            dm = HFDataManager(cfg)
            snapshot_seg = dm.snapshot
            trades_seg = dm.trades
            
            # 保存数据对
            data_pairs.append({
                'duration_ms': segment['duration_ms'],
                'segment_id': segment['segment_id'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'dm': dm,
                'snapshot': snapshot_seg,
                'trades': trades_seg
            })
            
            print(f"✓ Snapshot 数据行数: {len(snapshot_seg.data_df)}")
            print(f"✓ Trades 数据行数: {len(trades_seg.data)}")
            
        except ValueError as e:
            print(f"✗ 读取失败: {e}")
        except Exception as e:
            print(f"✗ 发生错误: {type(e).__name__}: {e}")
    
    print(f"\n{'='*60}")
    print(f"成功加载 {len(data_pairs)}/{len(time_segments)} 对数据")
    print(f"{'='*60}\n")
    
    return data_pairs

# # 使用示例
# # 步骤1：生成时间段
# time_segments = generate_time_segments(
#     start_time='2025-01-05 00:00:05.300',
#     durations=[100, 200, 150],
#     segments_per_duration=2,
#     gap_ms=50
# )

# # 步骤2：加载数据
# data_pairs = load_data_pairs(
#     time_segments=time_segments,
#     instrument="BTCUSDT",
#     root_type="data_share"
# )

# # 步骤3：使用数据
# print("数据对列表：")
# for i, pair in enumerate(data_pairs):
#     print(f"{i}: {pair['duration_ms']}ms #{pair['segment_id']} - "
#             f"{pair['start_time']} to {pair['end_time']}")

def evaluate_expressions(dm, expr1, expr2, period=slice(0, 1)):
    """
    评估两个表达式并返回结果
    
    参数:
        dm: HFDataManager 数据对象
        expr1: 第一个表达式（如 TradeVWAP）
        expr2: 第二个表达式（如 TradeAmount）
        period: 评估周期，默认为 slice(0, 1)
    
    返回:
        tuple: (expr1_result, expr2_result)
    """
    result1 = expr1.evaluate(dm, period=period)
    result2 = expr2.evaluate(dm, period=period)
    
    return result1, result2


# # 使用示例
# expr_vwap = TradeVWAP(100)
# expr_amount = TradeAmount(100)

# vwap_result, amount_result = evaluate_expressions(
#     dm=dm,
#     expr1=expr_vwap,
#     expr2=expr_amount
# )
