import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from pathlib import Path
from datetime import date
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict


from src.models import ObstructionDetectionModel
from src.data import generate_mine_set

log = logging.getLogger(__name__)

class Config:
    DEVICE = DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def test_batch(model, data):
    model.eval()
    filename, sample = data
    sample = sample.to(Config.DEVICE)
    outputs = model(sample)
    sigmoid = nn.Sigmoid()
    logits = sigmoid(outputs)
    predicted = (logits > 0.5).float()
    return list(filename), predicted, logits


def lightning_to_pytorch_state_conversion(state_dict):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if "criterion" in key:
            continue
        new_key = key[16:]
        new_state_dict[new_key] = value
    return new_state_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--batchsize", required=True, type=int)
    parser.add_argument("--csvfile", required=True, type=str)
    parser.add_argument("--fast_dev_run", type=int, default=1)
    args = parser.parse_args()
    return vars(args)


def main() -> None:
    args = parse_args()
    config = args["config"]
    point = args["model_checkpoint"]
    batch_size = args["batchsize"]
    csvfile = args["csvfile"]
    fast_dev_run = args["fast_dev_run"]

    PATH = Path.cwd()

    with open(config, "r") as stream:
        config = yaml.safe_load(stream)
    config = config["config"]
    labels = config["labels"]

    model = ObstructionDetectionModel(config)
    checkpoint = torch.load(point, weights_only=True, map_location=Config.DEVICE)
    _state_dict = lightning_to_pytorch_state_conversion(checkpoint["state_dict"])
    model.load_state_dict(_state_dict)
    model = model.to(Config.DEVICE)
    model.eval()

    predicted_values = []
    filenames = []
    logit_dict = {}

    for l in labels:
        logit_dict[l] = []

    set_dataloader = generate_mine_set(csvfile, batch_size, fast_dev_run)
    for bx, data in tqdm(enumerate(set_dataloader), total=len(set_dataloader)):
        filename, predicted, logits = test_batch(model, data)
        predicted_values.extend(predicted)
        for i, l in enumerate(labels):
            logit_dict[l].extend([x[i].item() for x in logits])
        filenames.extend(filename)
    predicted_values = [str(x.tolist()) for x in predicted_values]
    filenames = [x for x in filenames]

    predicted_values = np.array(predicted_values)
    filenames = np.array(filenames)

    df = pd.DataFrame(columns=["image_key", "labels","class"])
    df["image_key"] = filenames
    df["labels"] = predicted_values
    for key, value in logit_dict.items():
        df[key] = value


    today = date.today()
    today = today.strftime("%Y%m%d")
    point_name = point.split("/")[-2]
    csv_suffix = csvfile.split("_")[-1].split(".")[0]
    save_path = PATH / f"rc_filter_{today}_{csv_suffix}_{point_name}.csv"

    df.to_csv(save_path, index=False)

    log.info("Done")


if __name__ == "__main__":
    main()
