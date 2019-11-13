from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from models.SampleEmbeddingNet import SampleEmbeddingNet
from utils.plotter import PlotIdxsMng
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class TrainCycle:
    def __init__(self, model: SampleEmbeddingNet,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 epochs: int,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 writer: SummaryWriter,
                 device: torch.device,
                 plot_idx_mng: PlotIdxsMng):

        self.plot_idx_mng = plot_idx_mng
        self.device = device
        self.writer = writer
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model

    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].to(self.device)
            else:

                data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.writer.add_scalar(
                "Train/Batch Loss", loss.item(), self.plot_idx_mng.get_plot_idx("Train/Batch Loss"))

            prediction = torch.max(output, 1)
            total += target.size(0)

            correct += np.sum(prediction[1].cpu().numpy()
                              == target.cpu().numpy())
        self.train_loss = total_loss / len(self.test_loader)
        self.train_acc = correct / total
        self.writer.add_scalar("Train/Loss", self.train_loss, self.plot_idx_mng.get_plot_idx("Train/Loss"))
        self.writer.add_scalar("Train/Accuracy", self.train_acc, self.plot_idx_mng.get_plot_idx("Train/Accuracy"))
        self.writer.add_scalar("Model/MainNetNorm", self.model.get_main_net_norm(),
                               self.plot_idx_mng.get_plot_idx("Model/MainNetNorm"))
        self.writer.add_scalar("Model/EmbedsFactor", self.model.get_embed_factor(),
                               self.plot_idx_mng.get_plot_idx("Model/EmbedsFactor"))
    def test(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar(
                    "Test/Batch Loss", loss.item(), self.plot_idx_mng.get_plot_idx("Test/Batch Loss"))
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += np.sum(prediction[1].cpu().numpy()
                                  == target.cpu().numpy())
        self.test_loss = total_loss / len(self.test_loader)
        self.test_acc = correct / total
        self.writer.add_scalar("Test/Loss", self.test_loss, self.plot_idx_mng.get_plot_idx("Test/Loss"))
        self.writer.add_scalar("Test/Accuracy", self.test_acc, self.plot_idx_mng.get_plot_idx("Test/Accuracy"))
        self.writer.add_scalar("Model/EmbedNorm", self.model.get_embeds_norm(), self.plot_idx_mng.get_plot_idx("Model/EmbedNorm"))
        self.writer.add_scalar("Model/MainNetNorm", self.model.get_main_net_norm(), self.plot_idx_mng.get_plot_idx("Model/MainNetNorm"))
        self.writer.add_scalar("Model/EmbedsFactor", self.model.get_embed_factor(), self.plot_idx_mng.get_plot_idx("Model/EmbedsFactor"))


    def run(self):
        for e in range(self.epochs):
            self.test()
            self.train()
            self.on_epoch_end()

    def on_cycle_begin(self):
        pass

    def on_cycle_end(self):
        pass

    def on_epoch_end(self):
        pass


class TrainStrategy:
    def __init__(self, meta_cycles_count: int, train_cycles: List[TrainCycle]):
        self.meta_cycles_count = meta_cycles_count
        self.train_cycles = train_cycles

    def run(self):
        for tIdx in range(self.meta_cycles_count):
            for train_cycle in self.train_cycles:
                train_cycle.on_cycle_begin()
                train_cycle.run()
                train_cycle.on_cycle_end()
