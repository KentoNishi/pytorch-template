import torch

learning_rates = ((0.02, 80), (0.002, 100))
num_epochs = 250
momentum = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1024
num_workers = 8
save_path = "./saves"
data_path = "./data"
