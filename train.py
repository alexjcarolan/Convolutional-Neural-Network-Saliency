#!/usr/bin/env python3
import time
import torch
import pickle
import argparse
import numpy as np
import torch.backends.cudnn
import matplotlib.pyplot as plt
from dataset import *
from evaluate import *
from visualize import *
from pathlib import Path
from torch import nn, optim
from torchvision import utils
from torch.nn import functional
from torchvision import transforms
from typing import Union, NamedTuple
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="CNN for saliency prediction", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs to train for")
parser.add_argument("--batch-size", default=128, type=int, help="Number of images in each batch")
parser.add_argument("--initial-learning-rate", default=0.3, type=float, help="Initial earning rate")
parser.add_argument("--final-learning-rate", default=0.0001, type=float, help="Final Learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--weight-decay", default=0.0005, type=float, help="Weight decay")
parser.add_argument("--log-directory", default=Path("logs"), type=Path, help="Log directory path")
parser.add_argument("--log-frequency", default=100, type=int, help="Log frequency in steps")
parser.add_argument("--val-frequency", default=50, type=int, help="Validation frequency in epochs")
parser.add_argument("--worker-count", default=cpu_count(), type=int, help="Number of worker processes to load data")

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    log_directory = get_log_directory(args)
    summary_writer = SummaryWriter(str(log_directory), flush_secs=5)
    # unpickles training and validation set
    train_dataset = Salicon("trains.pkl")
    val_dataset = Salicon("vals.pkl")
    # loads training and validation set
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=args.worker_count)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, num_workers=args.worker_count)
    # initializes cnn and learning rates
    model = CNN(height=96, width=96, channels=3)
    learning_rates = np.linspace(args.initial_learning_rate, args.final_learning_rate, args.epochs + 1)
    # specifies critereion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # initializes trainer and begins training
    trainer = Trainer(model, train_loader, val_loader, learning_rates, criterion, optimizer, summary_writer, DEVICE)
    trainer.train(epochs=args.epochs, log_frequency=args.log_frequency, val_frequency=args.val_frequency)

    summary_writer.close()

def get_log_directory(args: argparse.Namespace):
    log_directory_prefix = (f"Salience_bs={args.batch_size}_ilr={args.initial_learning_rate}_flr={args.final_learning_rate}_m={args.momentum}_wd={args.weight_decay}_run_")
    i = 0
    log_directory = args.log_directory / (log_directory_prefix + str(i))
    while log_directory.exists():
        i += 1
        log_directory = args.log_directory / (log_directory_prefix + str(i))
    print(f"Writing logs to {log_directory}")
    return str(log_directory)

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        # first convolutional layer and max pooling
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(5, 5), padding=(2, 2))
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # second convolutional layer and max pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # third convolutional layer and max pooling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.initialise_layer(self.conv2)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # first fully connected layer
        self.fc1 = nn.Linear(15488, 4608)
        self.initialise_layer(self.fc1)
        # second fully conndected layer
        self.fc2 = nn.Linear(2304, 2304)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.conv1(images))
        x = self.pool1(x)
        x = functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = functional.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        x = self.maxout(x)
        x = self.fc2(x)
        return x
    # initialization of bias and weight
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, val=0.1)
        if hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
    # maxout layer of two even slices
    @staticmethod
    def maxout(tensor: torch.Tensor):
        slice = int(tensor.size(dim=1)/2)
        return tensor[:, :slice].max(tensor[:, slice:])

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, learning_rates: np.array, criterion: nn.Module, optimizer: Optimizer, summary_writer: SummaryWriter, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rates = learning_rates
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(self, epochs: int, log_frequency: int, val_frequency: int):
        self.model.train()
        for epoch in range(epochs):
            self.model.train()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                # predictions and loss calculated
                preds = self.model.forward(batch)
                loss = self.criterion(preds, labels)
                loss.backward()
                # loss backpropogated using the optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()
                # loss logged according to log frequency
                if ((self.step + 1) % log_frequency) == 0:
                    self.summary_writer.add_scalars("Loss", {"Train": loss}, self.step)

                self.step += 1
            # evaluation performed according to val frequency
            if ((epoch + 1) % val_frequency) == 0:
                self.evaluate(epoch)
                self.model.train()
            # learning rate linearly reduced
            for group in self.optimizer.param_groups:
                group['lr'] = self.learning_rates[epoch + 1]  
        # convolutional filters visualized
        filters = self.model.conv1.weight.data.clone()
        grid = utils.make_grid(filters, nrow=8, normalize=True, padding=1)
        plt.figure(figsize=(8, filters.shape[0] // 8 + 1 ))
        plt.axis('off')
        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.savefig("outputs/filters.png", bbox_inches='tight', pad_inches=0)
        # predicted saliency maps visualized
        visualize("preds.pkl", "vals.pkl", "outputs")

    def evaluate(self, epoch):
        total_loss = 0
        total_preds = []
        self.model.eval()
        # predictions, loss and accuracy calculated
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(batch)
                loss = self.criterion(preds, labels)
                total_loss += loss.item()
                total_preds.extend(list(preds.cpu().numpy()))
        
        average_loss = total_loss / len(self.val_loader)
        pickle.dump(total_preds, open("preds.pkl", "wb" ))
        cc, auc_shuffled, auc_borji = evaluate("preds.pkl", "vals.pkl")
        # loss and accuracy logged 
        self.summary_writer.add_scalars("Loss", {"Validation": average_loss}, self.step)
        self.summary_writer.add_scalars("Accuracy", {"CC": cc, "AUC Shuffled": auc_shuffled, "AUC Borji": auc_borji}, self.step)
        print(f"Epoch: [{epoch}] Loss: {average_loss:.5f} CC: {cc:.2f} AUC Shuffled: {auc_shuffled:.2f} AUC Borji: {auc_borji:.2f}")

if __name__ == "__main__":
    main(parser.parse_args())