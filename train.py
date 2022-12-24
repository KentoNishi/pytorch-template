import os

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import SGD

# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import *
import vars as v
import json


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
    # v.lr_scheduler.step(accuracy)
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
        v.model.parameters(), lr=args.learning_rates[0][0], momentum=args.momentum
    )
    v.current_epoch = 1
    # v.lr_scheduler = ReduceLROnPlateau(v.optimizer, "max")
    v.criterion = torch.nn.CrossEntropyLoss()
    os.makedirs(f"{args.save_path}/{tag}", exist_ok=True)
    v.writer = SummaryWriter(log_dir=f"{args.save_path}/{tag}")
    with open(f"{args.save_path}/{tag}/{tag}.json", "w+") as f:
        vs = vars(args)
        json.dump(
            {
                k: vs[k]
                for k in vs
                if not k.startswith("_") and not hasattr(vs[k], "__dict__")
            },
            f,
            indent=2,
            default=lambda o: str(o),
        )

    current_epoch_index = 0
    while v.current_epoch <= args.num_epochs:
        if v.current_epoch > args.learning_rates[current_epoch_index][1]:
            current_epoch_index += 1
        for p in v.optimizer.param_groups:
            p["lr"] = args.learning_rates[current_epoch_index][0]
        train()
        for key, value in test().items():
            v.writer.add_scalar(key, value, v.current_epoch)
        torch.save(
            {
                "epoch": v.current_epoch,
                "model_state_dict": v.model.state_dict(),
                "optimizer_state_dict": v.optimizer.state_dict(),
            },
            f"{args.save_path}/{tag}/{tag}.pt",
        )
        v.current_epoch += 1


if __name__ == "__main__":
    globals()[config]()
    loop()
