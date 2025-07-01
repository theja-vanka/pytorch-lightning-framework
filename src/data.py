import cv2
import torch
import pandas as pd
import lightning as L
import multiprocessing
from pathlib import Path
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


class CSVDataSet(Dataset):
    def __init__(self, df, root_path, file_column, label_column, transforms=None):
        super().__init__()
        self.df = pd.read_csv(df)
        self.transforms = transforms
        if root_path:
            self.root_path = Path(root_path)
        else:
            self.root_path = None
        self.file_column = file_column
        self.label_column = label_column


    def __getitem__(self, i):
        if self.root_path:
            image_path = self.root_path / self.df.iloc[i][self.file_column]
        else:
            image_path = self.df.iloc[i][self.file_column]
        sample = cv2.imread(image_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        label = eval(self.df.iloc[i][self.label_column])
        return sample, label

    def __len__(self):
        return len(self.df)

    def _collate_fn(self, batch):
        imgs, classes = list(zip(*batch))
        if self.transforms:
            imgs = [self.transforms(image=img)["image"][None] for img in imgs]
        classes = [torch.tensor([clss]) for clss in classes]
        imgs, classes = [torch.cat(i) for i in [imgs, classes]]
        return imgs, classes


class CSVInferenceDataSet(Dataset):
    def __init__(self, df, root_path, file_column, label_column, transforms=None):
        super().__init__()
        self.df = pd.read_csv(df)
        self.transforms = transforms
        if root_path:
            self.root_path = Path(root_path)
        else:
            self.root_path = None
        self.file_column = file_column
        self.label_column = label_column


    def __getitem__(self, i):
        if self.root_path:
            image_path = self.root_path / self.df.iloc[i][self.file_column]
        else:
            image_path = self.df.iloc[i][self.file_column]
        sample = cv2.imread(image_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        label = eval(self.df.iloc[i][self.label_column])
        return str(image_path), sample, label

    def __len__(self):
        return len(self.df)

    def _collate_fn(self, batch):
        filename, imgs, classes = list(zip(*batch))
        if self.transforms:
            imgs = [self.transforms(image=img)["image"][None] for img in imgs]
        classes = [torch.tensor([clss]) for clss in classes]
        imgs, classes = [torch.cat(i) for i in [imgs, classes]]
        filename = [str(f) for f in filename]
        return filename, imgs, classes


def generate_pseudo_set(df, root_path, file_column, label_column, batch_size):
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    test_dataset = CSVInferenceDataSet(df, root_path, file_column, label_column, transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset._collate_fn,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
    )
    return test_loader


class CSVMineDataSet(Dataset):
    def __init__(self, df, transforms=None, fast_dev_run=True):
        super().__init__()
        self.df = pd.read_csv(df)
        if fast_dev_run:
            self.df = self.df.head(10000)
        self.transforms = transforms

    def __getitem__(self, i):
        image_path = self.df.iloc[i].image_key
        sample = cv2.imread(image_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        return str(image_path), sample

    def __len__(self):
        return len(self.df)

    def _collate_fn(self, batch):
        filename, imgs = list(zip(*batch))
        if self.transforms:
            imgs = [self.transforms(image=img)["image"][None] for img in imgs]
        imgs = torch.cat(imgs)
        filename = [str(f) for f in filename]
        return filename, imgs


def generate_mine_set(filename, batch_size, fast_dev_run=1):
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    fast_dev_run = True if fast_dev_run == 1 else False
    mine_dataset = CSVMineDataSet(filename, transform, fast_dev_run)

    mine_loader = DataLoader(
        mine_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mine_dataset._collate_fn,
        num_workers=multiprocessing.cpu_count() - 1,
        pin_memory=True,
    )
    return mine_loader


class TFFCDataModule(L.LightningDataModule):
    def __init__(self, config, batch_size=None):
        super().__init__()
        self.config = config
        if batch_size is None:
            self.batch_size = self.config["batch_size"]
        else:
            self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        self.train_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
        self.train_dataset = CSVDataSet(
            self.config["train_df"], self.config["root_path"], self.config["file_column"], self.config["label_column"], self.train_transform
        )
        self.val_dataset = CSVDataSet(
            self.config["val_df"], self.config["root_path"], self.config["file_column"], self.config["label_column"], self.transform
        )
        self.test_dataset = CSVDataSet(
            self.config["test_df"], self.config["root_path"], self.config["file_column"], self.config["label_column"], self.transform
        )

    # multiprocessing.cpu_count() - 1
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset._collate_fn,
            num_workers=multiprocessing.cpu_count() - 1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset._collate_fn,
            num_workers=multiprocessing.cpu_count() - 1,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset._collate_fn,
            num_workers=multiprocessing.cpu_count() - 1,
            pin_memory=True,
        )


if __name__ == "__main__":
    pass