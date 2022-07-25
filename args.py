import torch

learning_rate = 0.02
num_epochs = 250
momentum = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1024
num_workers = 8
