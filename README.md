# AlphaHF

AlphaHF 是一个面向高频市场数据的因子生成与表达式搜索原型项目。项目的目标是把盘口快照 `snapshot` 与逐笔成交 `trades` 放进同一套表达式计算框架里，用统一时间轴完成特征构造、IC 评估和基于 RL 的表达式搜索。

当前版本是研究代码经过简单调整后得到的，所以更像一个研究快照。当前版本是single-asset 模式：一次只处理一个交易标的，例如 `BTCUSDT`。

## 项目亮点

- 统一管理两类高频数据：25 档盘口快照 + 逐笔成交
- 在 snapshot 时间轴上对 trades 做窗口聚合，解决两类数据时间粒度不一致的问题
- 基于 AlphaGen 风格表达式树扩展出更适合高频场景的算子集合
- 支持用 IC / Rank IC 评估表达式，并结合 RL 环境搜索候选 alpha

## 核心能力

### 1. Snapshot + Trades 的统一时间轴对齐

项目支持同时加载：

- 25 档盘口快照数据
- 逐笔成交数据

核心做法是以 snapshot 时间戳为主时间轴，把 trades 聚合到固定 lookback 窗口上，再作为表达式引擎中的一类可组合特征暴露出来。

相关实现：

- [data/alphaHF_data.py](data/alphaHF_data.py)
- [data/trades_aggregator.py](data/trades_aggregator.py)

一个简化示例：

```python
from data.alphaHF_data import HFConfig, HFDataManager
from data.snapshot_expression import TradeAmount, TradeVWAP

cfg = HFConfig(
    instrument="BTCUSDT",
    start_time="2025-01-05 00:00:05.300",
    end_time="2025-01-05 00:00:06.300",
    root_type="data_share",
    subsample="100ms",
    max_backtrack_ms=0,
    max_future_ms=0,
)

dm = HFDataManager(cfg)
trade_amount_100ms = TradeAmount(100).evaluate(dm, period=slice(0, 1))
trade_vwap_500ms = TradeVWAP(500).evaluate(dm, period=slice(0, 1))
```

这表示在 snapshot 时间轴上，分别取过去 `100ms` 和 `500ms` 的逐笔成交窗口，输出对应的成交额和 VWAP。

### 2. 面向高频研究的表达式系统

表达式引擎支持把市场特征、常数、时序窗口和算子组合成树结构表达式，并做批量张量化计算。

当前仓库中的算子能力覆盖：

- 基础算术与变换
- Rolling 时序统计
- 非线性激活类变换
- 分位数裁剪 / winsorization
- rank 与 rank-based 组合
- 双序列 rolling 关系建模
- 逐笔成交聚合因子，目前实现了 `TradeAmount` 和 `TradeVWAP`

相关实现：

- [data/snapshot_expression.py](data/snapshot_expression.py)
- [data/tree.py](data/tree.py)
- [data/snapshot_tokens.py](data/snapshot_tokens.py)

### 3. 因子评估与 RL 搜索

项目包含：

- 基于 IC / Rank IC 的表达式评估逻辑
- alpha pool 组合评估
- 用于表达式生成的 RL 环境与策略网络

相关实现：

- [data/calculator.py](data/calculator.py)
- [models/alpha_pool.py](models/alpha_pool.py)
- [rl/env/](rl/env/)
- [rl/policy.py](rl/policy.py)

## 仓库结构

```text
.
├── data/
│   ├── alphaHF_data.py          # snapshot / trades 数据加载与统一管理
│   ├── trades_aggregator.py     # trades 对齐到 snapshot 时间轴的聚合逻辑
│   ├── snapshot_expression.py   # 表达式系统与高频算子实现
│   ├── snapshot_tokens.py       # 表达式 token 定义
│   └── tree.py                  # 表达式树构建
├── models/
│   └── alpha_pool.py            # alpha pool 与组合评估
├── rl/
│   ├── env/
│   └── policy.py                # RL 环境与策略网络
├── verification/                # 验证脚本，目录名沿用历史命名
├── demo.py                      # 训练入口原型
├── config.py                    # 搜索空间与环境配置
└── utils.py                     # IC / Rank IC 等基础工具函数
```

## 安装

建议使用 Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据格式假设

当前代码默认读取本地目录中的 Binance Futures 高频数据，目录结构大致如下：

```text
<DATA_ROOT>/
└── BTCUSDT/
    ├── book_snapshot_25/
    │   └── binance-futures_book_snapshot_25_2025-01-01_BTCUSDT.csv.gz
    └── trades/
        └── binance-futures_trades_2025-01-01_BTCUSDT.csv.gz
```

其中：

- `book_snapshot_25` 用于读取盘口快照
- `trades` 用于读取逐笔成交

当前 `root_type` 解析逻辑支持以下取值：

- `data_share`
- `local`
- `data_local`

其中 `local` 和 `data_local` 都会映射到本地 `data_local/` 目录。

也就是说，如果直接运行现有代码，需要你本地已经准备好对应目录和数据文件。

## 最小使用示例

下面这个例子展示如何组合 snapshot 因子与 trades 因子：

```python
from data.alphaHF_data import HFConfig, HFDataManager, FeatureType
from data.snapshot_expression import Feature, TradeVWAP

cfg = HFConfig(
    instrument="BTCUSDT",
    start_time="2025-01-01",
    end_time="2025-01-02",
    root_type="data_share",
    subsample="100ms",
)

dm = HFDataManager(cfg)
mid_px = (Feature(FeatureType.ASK_1) + Feature(FeatureType.BID_1)) / 2
trade_vwap_1s = TradeVWAP(1000)
expr = (trade_vwap_1s / mid_px) - 1

result = expr.evaluate(dm, period=slice(0, 10))
```

参考脚本：

- [verification/feature_merge_script.py](verification/feature_merge_script.py)

## 验证脚本

仓库中包含用于和 Pandas 基准实现对比的验证脚本，主要用于研究阶段的正确性检查：

- `TradeAmount`
- `TradeVWAP`

相关文件：

- [verification/expr_pandas_compare.py](verification/expr_pandas_compare.py)
- [verification/validation_data_generator.py](verification/validation_data_generator.py)


## 训练入口

当前训练入口原型为：

- [demo.py](demo.py)
