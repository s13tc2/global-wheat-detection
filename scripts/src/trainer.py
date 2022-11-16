import os
import re
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import KFold

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

cfg = {
    "DIR_INPUT": "./data/",
    "DIR_TRAIN": f"./data/train",
    "DIR_TEST": f"./data/test",
    "num_epochs": 5,
    "use_amp": True,
    "model_file": "best_loss_min.pth",
    "detection_threshold": 0.5,
    "log_dir": "./logs/",
}

os.makedirs(cfg["log_dir"], exist_ok=True)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: DataLoader,
        loader_valid: DataLoader,
        batch_size: int,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.batch_size = batch_size
        self.train_loss = []
        self.loss_min = np.inf
        self.best_metric_model_file = "best_snapshot.pth"
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"logs.txt")
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=0.005, momentum=0.9, weight_decay=0.0005
        )
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 10
        )
        self.scaler = torch.cuda.amp.GradScaler() if cfg["use_amp"] else None
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_train_batch(self, images, targets):
        self.optimizer.zero_grad()

        with amp.autocast():
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        self.train_loss.append(loss.item())
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _run_train_epoch(self, epoch):
        # b_sz = len(next(iter(self.loader_train))[0])
        b_sz = self.batch_size
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.loader_train)}"
        )
        self.loader_train.sampler.set_epoch(epoch)
        for images, targets, image_ids in self.loader_train:
            images = list(image.to(self.gpu_id) for image in images)
            targets = [{k: v.to(self.gpu_id) for k, v in t.items()} for t in targets]

            self._run_train_batch(images, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _save_metric_best(self, metric):
        if metric > self.loss_min:
            print(
                f"metric_best ({self.loss_min:.6f} --> {metric:.6f}). Saving model ..."
            )
            torch.save(self.model.state_dict(), self.best_metric_model_file)
            self.loss_min = metric

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run + 1, max_epochs + 1):
            self.scheduler_cosine.step(epoch - 1)
            self._run_train_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            content = (
                time.ctime()
                + " "
                + f'Fold 0, Epoch {epoch}, lr: {self.optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(self.train_loss):.5f}.'
            )
            print(content)

            with open(self.log_file, "a") as appender:
                appender.write(content + "\n")

            metric = np.mean(self.train_loss)
            self._save_metric_best(metric)


def load_data(cfg):
    train_df = pd.read_csv(f'{cfg["DIR_INPUT"]}/train.csv')
    bboxs = np.stack(train_df["bbox"].apply(lambda x: np.fromstring(x[1:-1], sep=",")))
    for i, column in enumerate(["x", "y", "w", "h"]):
        train_df[column] = bboxs[:, i]
    train_df.drop(columns=["bbox"], inplace=True)
    kf = KFold(5)
    train_df["fold"] = -1
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df, train_df)):
        train_df.loc[valid_idx, "fold"] = fold

    return train_df


class WheatDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        super().__init__()

        self.image_ids = df["image_id"].unique()
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]

        image = cv2.imread(f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[["x", "y", "w", "h"]].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target['masks'] = None
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            sample = {"image": image, "bboxes": target["boxes"], "labels": labels}
            sample = self.transforms(**sample)
            image = sample["image"]

            target["boxes"] = torch.stack(
                tuple(map(torch.tensor, zip(*sample["bboxes"])))
            ).permute(1, 0)

        return image, target, image_id


def get_train_transform():
    return A.Compose(
        [A.Flip(0.5), ToTensorV2(p=1.0)],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def get_valid_transform():
    return A.Compose(
        [ToTensorV2(p=1.0)],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def prepare_dataloader_train(dataset: Dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        sampler=DistributedSampler(dataset),
    )


def prepare_dataloader_valid(dataset: Dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        sampler=DistributedSampler(dataset),
    )


def load_train_objs(cfg):

    train_df = load_data(cfg)

    fold = 0
    train_ = train_df[train_df["fold"] != fold].reset_index(drop=True)
    valid_ = train_df[train_df["fold"] == fold].reset_index(drop=True)
    dataset_train = WheatDataset(train_, cfg["DIR_TRAIN"], get_train_transform())
    dataset_valid = WheatDataset(valid_, cfg["DIR_TRAIN"], get_valid_transform())

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (wheat) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return dataset_train, dataset_valid, model


def ddp_setup():
    init_process_group(backend="nccl")


def main(
    save_every: int,
    total_epochs: int,
    batch_size: int,
    snapshot_path: str = "snapshot.pt",
):
    ddp_setup()
    (dataset_train, dataset_valid, model) = load_train_objs(cfg)

    loader_train = prepare_dataloader_train(dataset_train, batch_size)
    loader_valid = prepare_dataloader_valid(dataset_valid, batch_size)

    trainer = Trainer(
        model,
        loader_train,
        loader_valid,
        batch_size,
        save_every,
        snapshot_path,
    )
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size", default=16, help="Input batch size on each device (default: 32)"
    )
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)
