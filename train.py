import os
import time
import torch
import mlflow
from watermark import watermark
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks import LearningRateMonitor

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.data import TFFCDataModule
from src.models import LightningModel

os.environ["LOGNAME"] = "krishnatheja.vanka"
torch.set_float32_matmul_precision("high")


class Config:
    MODEL_NAME = "class3"


def cli_main():
    version_no = time.strftime("%Y%m%d_%H%M%S", time.gmtime(int(float(time.time()))))
    model_name = Config.MODEL_NAME
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"artifacts/iter_{version_no}_{model_name}/",
        filename="RC_{val_ts_date:.12g}_{val_ts_time:g}_ep{epoch:03d}",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )
    mlflow.set_experiment(experiment_name="tffc-road-condition-multi-label")
    mlflow.start_run(run_name=f"{version_no}")
    mlflow.pytorch.autolog()
    cli = LightningCLI(
        LightningModel,
        TFFCDataModule,
        seed_everything_default=42,
        run=False,
        trainer_defaults={
            "fast_dev_run": False,
            "callbacks": [
                checkpoint_callback,
                RichProgressBar(),
                LearningRateFinder(),
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(monitor="train_acc", mode="max"),
            ],
        },
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    mlflow.end_run()


if __name__ == "__main__":
    # python train.py --config conf/masterconfig.yaml
    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())
    cli_main()
