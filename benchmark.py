import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from argparse import ArgumentParser
from collections import OrderedDict

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_curve

from src.models import ObstructionDetectionModel
from src.data import generate_pseudo_set

log = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Config:
    DEVICE = DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1024
    DATASET_NAMES = ["train", "val", "test"]
    TRAIL_NAME = None


@torch.inference_mode()
def test_batch(model, data):
    model.eval()
    filename, sample, target = data
    target = target.float()
    sample = sample.to(Config.DEVICE)
    target = target.to(Config.DEVICE)
    outputs = model(sample)
    sigmoid = nn.Sigmoid()
    logits = sigmoid(outputs)
    predicted = (logits > 0.5).float()
    return predicted, target, list(filename), logits


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
    args = parser.parse_args()
    return vars(args)


def create_inference_csv(set_dataloader, model, lst_labels):
    predicted_values = []
    target_values = []
    filenames = []
    logits_values = []

    for bx, data in tqdm(enumerate(set_dataloader), total=len(set_dataloader)):
        predicted, target, filename, logits = test_batch(model, data)
        predicted_values.extend(predicted)
        target_values.extend(target)
        filenames.extend(filename)
        logits_values.extend(logits)

    predicted_values = [str(x.tolist()) for x in predicted_values]
    target_values = [str(x.tolist()) for x in target_values]
    filenames = [x for x in filenames]
    logits_values = [x.cpu().detach().numpy() for x in logits_values]

    predicted_values = np.array(predicted_values)
    target_values = np.array(target_values)
    filenames = np.array(filenames)
    logits_values = np.array(logits_values)

    df = pd.DataFrame(
        columns=[
            "filename",
            "target_values",
            "predicted_values",
        ]
    )
    df["filename"] = filenames
    df["target_values"] = target_values
    df["predicted_values"] = predicted_values
    # df["target_values"] = df["target_values"].astype(int)
    # df["predicted_values"] = df["predicted_values"].astype(list)

    for index, value in enumerate(lst_labels):
        df[value] = logits_values[:, index]

    return df


def create_confusion_matrix(
    cf_matrix, individual_matrix, label_name, metric_save_path
):
    _save_cf_matrix = pd.DataFrame(
        individual_matrix,
        columns=[
            "NO_" + label_name,
            label_name,
        ],
        index=[
            "NO_" + label_name,
            label_name,
        ],
    )
    plt.figure()
    cf_plot = sns.heatmap(
        individual_matrix, annot=True, fmt="d", cbar=False, cmap="coolwarm"
    )
    cf_plot.set(xlabel="Predicted", ylabel="Actual")
    cf_plot.set_xticklabels(
        labels=[
            "NO_" + label_name,
            label_name,
        ],
        rotation=90,
    )
    cf_plot.set_yticklabels(
        labels=[
            "NO_" + label_name,
            label_name,
        ],
        rotation=0,
    )
    fig = cf_plot.get_figure()
    _save_cf_matrix.to_csv(metric_save_path / f"confusion_matrix_{label_name}.csv")
    fig.savefig(metric_save_path / f"confusion_matrix_{label_name}.jpg", bbox_inches="tight")

    return _save_cf_matrix


def create_benchmark_csv(_save_cf_matrix, label_name, individual_matrix, metric_save_path):
    _save_cf_matrix.loc["predicted"] = _save_cf_matrix.sum(axis=0)
    _save_cf_matrix["ground_truth"] = _save_cf_matrix.sum(axis=1)
    _ground_truth = _save_cf_matrix["ground_truth"].values.tolist()[:-1]
    _predicted = _save_cf_matrix.loc["predicted"].values.tolist()[:-1]
    metric_table = pd.DataFrame(
        [_ground_truth, _predicted],
        columns=[
            "NO_" + label_name,
            label_name,
        ],
        index=["GT", "Pred"],
    )

    _tp = []
    for index in range(2):
        _tp.append(individual_matrix[index][index])

    metric_table.loc["TP"] = _tp
    metric_table.loc["FP"] = metric_table.loc["Pred"] - metric_table.loc["TP"]
    metric_table.loc["FN"] = metric_table.loc["GT"] - metric_table.loc["TP"]
    metric_table.loc["Prec"] = metric_table.loc["TP"] / (
        metric_table.loc["TP"] + metric_table.loc["FP"]
    )
    metric_table.loc["Recall"] = metric_table.loc["TP"] / (
        metric_table.loc["TP"] + metric_table.loc["FN"]
    )
    metric_table.loc["F1_score"] = (
        2
        * (metric_table.loc["Prec"] * metric_table.loc["Recall"])
        / (metric_table.loc["Prec"] + metric_table.loc["Recall"])
    )
    metric_table.loc["Accuracy"] = (
        metric_table.loc["TP"].sum() / metric_table.loc["GT"].sum()
    )
    metric_table = metric_table.T
    metric_table.to_csv(metric_save_path / f"metrics_{label_name}.csv")


def create_pr_curve(df, lst_labels, curves_save_path, save_name):
    y_target = np.array(
        [eval(multi_label) for multi_label in df["target_values"].values.tolist()]
    )
    y_pred = df[lst_labels].values

    pr_lst = []
    desired_intervals = np.arange(0, 1.1, 0.1)
    for index, value in enumerate(lst_labels):
        pr_df = pd.DataFrame(
            columns=[
                "class",
                "thresholds",
                "precision",
                "recall",
            ]
        )
        precision, recall, thresholds = precision_recall_curve(
            y_target[:, index], y_pred[:, index], drop_intermediate=True
        )
        closest_indices = np.argmin(
            np.abs(thresholds[:, None] - desired_intervals), axis=0
        )

        precision_intervals = precision[closest_indices]
        recall_intervals = recall[closest_indices]

        desired_intervals = np.array(desired_intervals, dtype="float16")
        precision_intervals = np.array(precision_intervals, dtype="float32")
        recall_intervals = np.array(recall_intervals, dtype="float32")

        pr_df["thresholds"] = desired_intervals
        pr_df["precision"] = precision_intervals
        pr_df["recall"] = recall_intervals
        pr_df["class"] = value

        _pr_save_name = str(value) + ".csv"
        pr_df.to_csv(curves_save_path / _pr_save_name, index=False)
        pr_lst.append(pr_df)
    pr_lst = pd.concat(pr_lst, ignore_index=True)
    plt.figure()
    pr_plot = sns.lineplot(pr_lst, x="recall", y="precision", hue="class")
    pr_plot.set_title("Precision vs. Recall Curve")
    fig = pr_plot.get_figure()
    save_name = save_name.rstrip(".jpg") + "pr.jpg"
    fig.savefig(curves_save_path / save_name, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    config = args["config"]
    point = args["model_checkpoint"]

    if not Config.TRAIL_NAME:
        folder_point = point.split("/")[-2]
    else:
        folder_point = point.split("/")[-2] + "_" + Config.TRAIL_NAME

    point = Path(point)
    result_path = Path("report/")

    with open(config, "r") as stream:
        config = yaml.safe_load(stream)

    with open("conf/" + config["model"], "r") as stream:
        model_config = yaml.safe_load(stream)

    data_config = config["data"]["config"]

    model = ObstructionDetectionModel(model_config["config"])
    checkpoint = torch.load(point, weights_only=True, map_location=Config.DEVICE)
    _state_dict = lightning_to_pytorch_state_conversion(checkpoint["state_dict"])
    model.load_state_dict(_state_dict)
    model = model.to(Config.DEVICE)

    for dset in Config.DATASET_NAMES:
        dataset = data_config[dset + "_df"]
        set_name = dataset.split("/")[-1].rstrip(".csv")

        set_dataloader = generate_pseudo_set(
            dataset,
            data_config["root_path"],
            data_config["file_column"],
            data_config["label_column"],
            Config.BATCH_SIZE,
        )
        lst_labels = model_config["config"]["labels"]

        save_name = Path(point).as_posix()
        save_name = save_name.split("/")[-1].replace(".ckpt", f"_{dset}.csv")

        df = create_inference_csv(set_dataloader, model, lst_labels)

        save_path = result_path / folder_point / set_name
        save_path.mkdir(exist_ok=True, parents=True)

        df.to_csv(save_path / save_name, index=False)

        metric_save_path = result_path / folder_point / set_name / "metrics"
        metric_save_path.mkdir(exist_ok=True, parents=True)

        _deserialized_target_values = [
            eval(multi_label) for multi_label in df["target_values"].values.tolist()
        ]
        _deserialized_predicted_values = [
            eval(multi_label) for multi_label in df["predicted_values"].values.tolist()
        ]
        cf_matrix = multilabel_confusion_matrix(
            _deserialized_target_values, _deserialized_predicted_values
        )
        for individual_matrix, label_name in zip(cf_matrix, lst_labels):
            _save_cf_matrix = create_confusion_matrix(
                cf_matrix, individual_matrix, label_name, metric_save_path
            )
            create_benchmark_csv(
                _save_cf_matrix, label_name, individual_matrix, metric_save_path
            )

        curves_save_path = result_path / folder_point / set_name / "pr_curves"
        curves_save_path.mkdir(exist_ok=True, parents=True)
        create_pr_curve(df, lst_labels, curves_save_path, save_name)

    log.info("Done")


if __name__ == "__main__":
    # python benchmark.py --model_checkpoint <checkpoint> --config conf/inference.yaml
    main()
