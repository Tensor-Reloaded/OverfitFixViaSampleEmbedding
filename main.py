# python main.py --lr=0.05 --lr_milestones 30 60 90 120 150 180 210 240 270 300 --lr_gamma=0.5 --wd=0.0005 --nesterov --momentum=0.9 --model="VGG('VGG11')" --epoch=300 --train_batch_size=128
import os
from math import sqrt

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from cyclic_schedulers.cos_annealing_lr_scheduler import ReversedCosLRScheduler,CosLRScheduler
import argparse

from models import *
from misc import progress_bar
from learn_utils import reset_seed
from utils.SampleAndIdxDataset import SampleIdxDataset

from utils.train_cycle_strategies import *
from utils.train_strategy import TrainStrategy

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--model', default="VGG('VGG19')",
                        type=str, help='what model to use')

    parser.add_argument('--half', '-hf', action='store_true',
                        help='use half precision')
    parser.add_argument('--load_model', default="",
                        type=str, help='what model to load')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.0,
                        type=float, help='sgd momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of epochs tp train for')
    parser.add_argument('--train_batch_size', default=128,
                        type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=512,
                        type=int, help='testing batch size')
    parser.add_argument('--initialization', '-init', default=0, type=int,
                        help='The type of initialization to be used \n 0 - Default pytorch initialization \n 1 - Xavier Initialization\n 2 - He et. al Initialization\n 3 - SELU Initialization\n 4 - Orthogonal Initialization')
    parser.add_argument('--initialization_batch_norm', '-init_batch',
                        action='store_true', help='use batch norm initialization')

    parser.add_argument('--save_model', '-save',
                        action='store_true', help='perform_top_down_sum')
    parser.add_argument('--save_interval', default=5,
                        type=int, help='perform_top_down_sum')
    parser.add_argument('--save_dir', default="checkpoints",
                        type=str, help='save dir name')

    parser.add_argument('--num_workers_train', default=4,
                        type=int, help='number of workers for loading train data')
    parser.add_argument('--num_workers_test', default=2,
                        type=int, help='number of workers for loading test data')

    parser.add_argument('--cuda', default=torch.cuda.is_available(),
                        type=bool, help='whether cuda is in use')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed to be used by randomizer')
    parser.add_argument('--progress_bar', '-pb',
                        action='store_true', help='Show the progress bar')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.args = config
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        if self.args.save_dir == "" or self.args.save_dir == None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/"+self.args.save_dir)
        self.batch_plot_idx = 0

    def load_data(self):
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = SampleIdxDataset(torchvision.datasets.CIFAR100(
            root='../storage', train=True, download=True, transform=train_transform))
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=self.args.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR100(
            root='../storage', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = eval(self.args.model)
        # self.embed_transform = EmbedUpsample(4)
        self.embed_transform = NoUpsample()
        self.model = SampleEmbeddingNet(self.model, embeddings_count=50000, embed_dim=3*32*32, embed_transform=self.embed_transform,embed_factor=1.0,embed_max_norm=None)

        self.save_dir = "../storage/" + self.args.save_dir
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        if self.cuda:
            if self.args.half:
                self.model.half()
                for layer in self.model.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.float()
                print("Using half precision")

        if self.args.initialization == 1:
            # xavier init
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform(
                        m.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.initialization == 2:
            # he initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal(m.weight, mode='fan_in')
        elif self.args.initialization == 3:
            # selu init
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.kernel_size[0] * \
                        m.kernel_size[1] * m.in_channels
                    nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
        elif self.args.initialization == 4:
            # orthogonal initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal(m.weight)

        if self.args.initialization_batch_norm:
            # batch norm initialization
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)

        if len(self.args.load_model) > 0:
            print("Loading model from " + self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))
        self.model = self.model.to(self.device)

        # self.optimizer = optim.SGD(self.model.parameters(
        # ), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        self.optimizer  =optim.SGD([
            {'params': self.model.main_net.parameters()},
            {'params': self.model.embed.parameters()},
        ], lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.wd)

        # self.optimizer = optim.SGD(self.model.embed.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)


        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def get_batch_plot_idx(self):
        self.batch_plot_idx += 1
        return self.batch_plot_idx - 1

    def train(self):
        print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, target, idxs) in enumerate(self.train_loader):
            data, target,idxs = data.to(self.device), target.to(self.device), idxs.to(self.device)
            if self.device == torch.device('cuda') and self.args.half:
                data = data.half()
            self.optimizer.zero_grad()

            output = self.model(data, idxs, self.embed_pen)
            # embed_weights = self.model.embed.weight.norm()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.writer.add_scalar(
                "Train/Batch Loss", loss.item(), self.get_batch_plot_idx())
            # second param "1" represents the dimension to be reduced
            prediction = torch.max(output, 1)
            total += target.size(0)

            correct += np.sum(prediction[1].cpu().numpy()
                              == target.cpu().numpy())

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (total_loss / (batch_num + 1), 100.0 * correct/total, correct, total))

        return total_loss, correct / total

    def test(self):
        print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target, idxs) in enumerate(self.test_loader):
                data, target, idxs = data.to(self.device), target.to(self.device), idxs.to(self.device)
                if self.device == torch.device('cuda') and self.args.half:
                    data = data.half()
                idxs = idxs + 1000000
                output = self.model(data, idxs)
                loss = self.criterion(output, target)
                self.writer.add_scalar(
                    "Test/Batch Loss", loss.item(), self.get_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)

                correct += np.sum(prediction[1].cpu().numpy()
                                  == target.cpu().numpy())

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss, correct/total

    def save(self, epoch, accuracy, tag=None):
        if tag != None:
            tag = "_"+tag
        else:
            tag = ""
        model_out_path = self.save_dir + \
            "/model_{}_{}{}.pth".format(
                epoch, accuracy * 100, tag)
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):

        ONLY_EMBEDS_EC = 5
        FULL_EC = 10
        ONLY_MAIN_NET_EC = 10
        MIN_LR_VAL = self.args.lr
        CYCLE_MUL = 1

        reset_seed(self.args.seed)
        self.load_data()
        self.load_model()



        plot_idx_mng = PlotIdxsMng()
        only_embeds_cos_scheduler = CosineAnnealingWarmRestarts(self.optimizer, ONLY_EMBEDS_EC, CYCLE_MUL, MIN_LR_VAL)
        full_model_cos_scheduler = CosineAnnealingWarmRestarts(self.optimizer, FULL_EC, CYCLE_MUL, MIN_LR_VAL)
        only_main_net_cos_scheduler = CosineAnnealingWarmRestarts(self.optimizer, ONLY_MAIN_NET_EC, CYCLE_MUL, MIN_LR_VAL)

        # only_embeds_cycle = TrainOnlyEmbeds(self.model, self.optimizer, self.criterion, ONLY_EMBEDS_EC, self.train_loader, self.test_loader, self.writer, self.device, plot_idx_mng,only_embeds_cos_scheduler)
        full_model_cycle = TrainFullModelZeroOutEmbeds(self.model, self.optimizer, self.criterion, FULL_EC, self.train_loader, self.test_loader, self.writer, self.device, plot_idx_mng, full_model_cos_scheduler)
        only_main_net_cycle = TrainOnlyMainNetZeroOutEmbedsFactor(self.model, self.optimizer, self.criterion, ONLY_MAIN_NET_EC, self.train_loader, self.test_loader, self.writer, self.device, plot_idx_mng,only_main_net_cos_scheduler)

        # ts = TrainStrategy(15, [only_embeds_cycle, full_model_cycle, only_main_net_cycle])
        # ts = TrainStrategy(15, [full_model_cycle, only_main_net_cycle])
        ts = TrainStrategy(15, [full_model_cycle])
        ts.run()



        # accuracy = 0
        # for epoch in range(1, self.args.epoch + 1):
        #     if epoch % 5 == 0:
        #         self.args.wd = max(0.0005, 0.7 * self.args.wd)
        #         self.embed_pen = max(0.0, self.embed_pen + self.args.embed_pen_inc)
        #         self.optimizer = optim.SGD([
        #             {'params': self.model.features.parameters()},
        #             {'params': self.model.classifier.parameters()},
        #             {'params': self.model.embed.parameters(), 'weight_decay': 0.0},
        #         ], lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        #
        #     print("\n===> epoch: %d/%d" % (epoch, self.args.epoch))
        #
        #     train_result = self.train()
        #     # if epoch == self.args.embed_decay_miletones[0]:
        #         # self.embed_pen += self.args.embed_pen_inc
        #         # print("INCREASING EMBEDDING PEN TO ",self.embed_pen)
        #         # self.args.embed_decay_miletones = self.args.embed_decay_miletones[1:]
        #     loss = train_result[0]
        #     accuracy = train_result[1]
        #
        #     self.writer.add_scalar("Train/Loss", loss, epoch)
        #     self.writer.add_scalar("Train/Accuracy", accuracy, epoch)
        #
        #     test_result = self.test()
        #
        #     loss = test_result[0]
        #     accuracy = test_result[1]
        #
        #     self.writer.add_scalar("Test/Loss", loss, epoch)
        #     self.writer.add_scalar("Test/Accuracy", accuracy, epoch)
        #
        #     self.writer.add_scalar("Model/Norm", self.get_model_norm(), epoch)
        #     self.writer.add_scalar(
        #         "Train Params/Learning rate", self.scheduler.get_lr()[0], epoch)
        #
        #     self.writer.add_scalar("Model/EmbedWeightsNorm", self.model.get_norm_of_weights_connected_to_embeds(), epoch)
        #     self.writer.add_scalar("Model/EmbedWeightsPen", self.embed_pen, epoch)
        #     self.writer.add_scalar("Model/EmbedNorm", self.model.embed.weight.norm(), epoch)
        #     #
        #     if accuracy < test_result[1]:
        #         accuracy = test_result[1]
        #         self.save(epoch, accuracy)
        #
        #     if self.args.save_model and epoch % self.args.save_interval == 0:
        #         self.save(0, epoch)
        #
        #     if self.args.use_reduce_lr:
        #         self.scheduler.step(train_result[0])
        #     else:
        #         self.scheduler.step(epoch)

    def get_model_norm(self, norm_type=2):
        norm = 0.0
        for param in self.model.parameters():
            norm += torch.norm(input=param, p=norm_type, dtype=torch.float)
        return norm


if __name__ == '__main__':
    main()
