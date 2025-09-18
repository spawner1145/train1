# python trainer.py --config config/test.yaml
import os
import sys

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

import torch
import lightning as pl

from common.utils import get_class, parse_args, create_scaled_precision_plugin
from common.trainer import Trainer
from omegaconf import OmegaConf
from lightning.fabric.connector import _is_using_cli

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.trainer.resume = args.resume
    plugins = []

    strategy = config.lightning.pop("strategy", "auto")
    strategy_params = config.lightning.pop("strategy_params", {})
    if "." in strategy:
        strategy = get_class(strategy)(**strategy_params)
    elif strategy == "ddp":
        from lightning.fabric.strategies import DDPStrategy
        strategy = DDPStrategy(**strategy_params)
    elif strategy == "fsdp":
        from lightning.fabric.strategies import FSDPStrategy
        strategy = FSDPStrategy(**strategy_params)
        
    elif strategy == "deepspeed":
        from lightning.fabric.strategies import DeepSpeedStrategy
        strategy = DeepSpeedStrategy(**strategy_params)
    loggers = pl.fabric.loggers.CSVLogger(".")
    
    if config.trainer.wandb_id != "":
        from lightning.pytorch.loggers import WandbLogger
        kwargs = dict(project=config.trainer.wandb_id)
        if config.trainer.get("wandb_entity", None):
            kwargs["entity"] = config.trainer.wandb_entity
        loggers = WandbLogger(**kwargs)
        
    if config.lightning.precision == "16-true-scaled":
        config.lightning.precision = None
        config._scaled_fp16_precision = True
        plugins.append(create_scaled_precision_plugin())

    fabric = pl.Fabric(
        loggers=[loggers], 
        plugins=plugins, 
        strategy=strategy, 
        **config.lightning
    )
    if not _is_using_cli():
        fabric.launch()
        
    fabric.barrier()
    fabric.seed_everything(config.trainer.seed)
    Trainer(fabric, config).train_loop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Captured KeyboardInterrupt, exiting...")
        sys.exit(0)