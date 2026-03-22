import logging
import multiprocessing
import random
import warnings

import numpy as np
import torch

from sb3_contrib import MaskablePPO

# Core AlphaHF modules
from models.alpha_pool import LinearAlphaPool
from data.alphaHF_data import HFConfig, HFDataManager, FeatureType
from data.calculator import CryptoAlphaCalculator
from data.snapshot_expression import Feature, Ref
from rl.env.wrapper import AlphaEnv
from rl.policy import LSTMSharedNet

warnings.filterwarnings("ignore", message=".*does not contain valid edges.*")

TRAIN_START = "2025-01-01"
TRAIN_END = "2025-01-02"

def set_seed(seed):
    """Set random seed to ensure reproducible results"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic convolution algorithm selection
    torch.backends.cudnn.benchmark = False     # Set to True if network input dimensions/types don't vary much

def setup_logging():
    # Remove all existing handlers to prevent duplication or overwriting
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for detailed logs
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()  # Get root logger
    return logger

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

def main():
    logger = setup_logging()
    logger.debug("Starting main training script.")

    # Device configuration, modify here to ensure correct GPU is used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------ Hyperparameter settings ------------------ #
    SEED = 1
    STEPS = 100_000  
    INSTRUMENT = "BTCUSDT"
    ROOT = "data_share"

    # ------------- Initialize data and targets ------------- #
    train_cfg = HFConfig(
        instrument=INSTRUMENT,
        start_time=TRAIN_START,
        end_time=TRAIN_END,
        root_type=ROOT,
        device=device,
    )
    data_train = HFDataManager(train_cfg)

    mid_px = (Feature(FeatureType.ASK_1) + Feature(FeatureType.BID_1)) / 2
    target = Ref(mid_px, -20) / mid_px - 1
    calculator_train = CryptoAlphaCalculator(dm=data_train, target=target)
    logger.info(f"Train data shape: {data_train.snapshot.data.shape}, n_days={data_train.n_ms}")
    logger.info(f"Mid price feature shape: {mid_px.evaluate(data_train).shape}")

    set_seed(seed=SEED)

    # 先创建 pool（用于 env 里计算 reward）
    pool = LinearAlphaPool(
        capacity=10,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=device
    )

    # 再创建环境，并传入 pool
    env = AlphaEnv(
        pool=pool,
        device=device,     
        print_expr=True
    )
    env.reset()
    logger.debug("Environment initialization completed.")

    policy_kwargs = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(
            n_layers=2,
            d_model=128,
            dropout=0.1,
            device=device,
        ),
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64, 
        gamma=1.0,     # 因为因子生成是一次性的，没有长远的“未来”，Gamma设为1
        device=device,
        tensorboard_log="./logs"
    )

    # ----------------------------------------
    # 开始训练 (Training Loop)
    # ----------------------------------------
    print("开始训练...")
    model.learn(total_timesteps=STEPS)
    
    # ----------------------------------------
    # 保存模型
    # ----------------------------------------
    model.save("ppo_crypto_alpha")
    print("训练结束，模型已保存。")

if __name__ == "__main__":
    main()
