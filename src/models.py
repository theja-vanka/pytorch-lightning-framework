import torch
import torch.nn as nn
import lightning.pytorch as L
from datetime import datetime
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MultilabelAccuracy


class Config:
    CRITERION = nn.BCEWithLogitsLoss()
    OPTIMIZER = torch.optim.AdamW
    SCHEDULER = CosineAnnealingLR


class ObstructionDetectionModel(nn.Module):
    def __init__(self, config):
        super(ObstructionDetectionModel, self).__init__()
        self.config = config
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V2")
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, self.config["classes"]
        )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


class LightningModel(L.LightningModule):
    def __init__(self, config, learning_rate=0.00001):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.config = config
        self.criterion = Config.CRITERION
        self.optimizer = Config.OPTIMIZER
        self.scheduler = Config.SCHEDULER

        self.model = ObstructionDetectionModel(config)
        self.model = torch.compile(self.model)

        self.train_acc = MultilabelAccuracy(num_labels=config["classes"])
        self.val_acc = MultilabelAccuracy(num_labels=config["classes"])
        self.test_acc = MultilabelAccuracy(num_labels=config["classes"])

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        sample, target = batch
        logits = self(sample)
        loss = self.criterion(logits, target.float())
        return loss, target, logits

    def on_train_epoch_end(self):
        _train_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_ts_date = int(_train_timestamp.split("_")[0])
        train_ts_time = int(_train_timestamp.split("_")[1])
        self.log("train_ts_date", train_ts_date)
        self.log("train_ts_time", train_ts_time)

    def on_validation_epoch_end(self):
        _val_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        val_ts_date = int(_val_timestamp.split("_")[0])
        val_ts_time = int(_val_timestamp.split("_")[1])
        self.log("val_ts_date", val_ts_date)
        self.log("val_ts_time", val_ts_time)

    def training_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.train_acc(logits, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, true_labels, logits = self._shared_step(batch)
        self.test_acc(logits, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        scheduler = self.scheduler(optimizer, T_max=5)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    pass
