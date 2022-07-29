from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

model: Optional[torch.nn.Module] = None
optimizer: Optional[torch.optim.Optimizer] = None
# lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
criterion: Optional[torch.nn.modules.loss._Loss] = None
trainloader: Optional[DataLoader] = None
testloader: Optional[DataLoader] = None
writer: Optional[SummaryWriter] = None
current_epoch: int = 0
