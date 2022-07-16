import argparse
import os

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import args
import configs
import vars as v


def train():
    v.model.train()
    t = tqdm(total=len(v.trainloader), desc=f"Train Epoch {v.current_epoch}")
    for batch_idx, (data, target) in enumerate(v.trainloader):
        data, target = data.to(args.device), target.to(args.device)
        v.optimizer.zero_grad()
        output = v.model(data)
        loss = v.criterion(output, target)
        loss.backward()
        v.optimizer.step()
        t.update(1)
        t.set_postfix({"Loss": loss.item()})
    t.close()


def test():
    v.model.eval()
    labels = []
    preds = []
    t = tqdm(total=len(v.testloader), desc=f"Test Epoch {v.current_epoch}")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(v.testloader):
            data, target = data.to(args.device), target.to(args.device)
            output = v.model(data)
            _, predicted = torch.max(output.data, 1)
            labels.extend(target.tolist())
            preds.extend(predicted.tolist())
            t.update(1)
    accuracy: float = accuracy_score(labels, preds)
    precision: float = precision_score(labels, preds, average="macro")
    recall: float = recall_score(labels, preds, average="macro")
    f1: float = f1_score(labels, preds, average="macro")
    v.lr_scheduler.step(accuracy)
    t.close()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def loop():
    v.model = v.model.to(args.device)
    v.optimizer = SGD(
        v.model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
    v.current_epoch = 1
    v.lr_scheduler = ReduceLROnPlateau(v.optimizer, "max")
    v.criterion = torch.nn.CrossEntropyLoss()
    v.writer = SummaryWriter(log_dir=f"./saves/{tag}")

    while v.current_epoch <= args.num_epochs:
        train()
        for key, value in test().items():
            v.writer.add_scalar(key, value, v.current_epoch)
        os.makedirs(f"./saves/{tag}", exist_ok=True)
        torch.save(
            {
                "epoch": v.current_epoch,
                "model_state_dict": v.model.state_dict(),
                "optimizer_state_dict": v.optimizer.state_dict(),
            },
            f"./saves/{tag}/{tag}.pt",
        )
        v.current_epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Experiment config name.",
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        required=True,
        help="Experiment tag name.",
    )
    cli_args = parser.parse_args()
    config, tag = cli_args.config, cli_args.tag
    configs.__dict__[config]()
    loop()
