import torch

from torch import nn

from cyclic_schedulers.cos_annealing_scheduler import CyclicCosAnnealingScheduler
from models.SampleEmbeddingNet import SampleEmbeddingNet
from utils.plotter import PlotIdxsMng
from utils.train_strategy import TrainCycle

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class TrainFullModel(TrainCycle):
    def __init__(self, model: SampleEmbeddingNet,
                 optimizer, criterion, epochs, train_loader, test_loader, writer,
                 device,
                 plot_idx_mng: PlotIdxsMng, scheduler):
        super().__init__(model, optimizer, criterion, epochs, train_loader, test_loader, writer, device, plot_idx_mng)
        self.scheduler = scheduler
        # self.scheduler.step()

    def on_cycle_begin(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def on_cycle_end(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def on_epoch_end(self):
        self.scheduler.step()
        self.writer.add_scalar("Train Params/Learning rate" , self.optimizer.param_groups[0]['lr'], self.plot_idx_mng.get_plot_idx("Train Params/Learning rate"))

class TrainFullModelZeroOutEmbeds(TrainFullModel):
    def __init__(self, model: SampleEmbeddingNet, optimizer, criterion, epochs, train_loader, test_loader, writer,
                 device, plot_idx_mng: PlotIdxsMng, scheduler):
        super().__init__(model, optimizer, criterion, epochs, train_loader, test_loader, writer, device, plot_idx_mng,
                         scheduler)

    def on_cycle_begin(self):
        super().on_cycle_begin()
        # self.model.embed.weight = nn.Parameter(torch.rand(self.model.embed.weight.size(),device='cuda'))
        # self.model.embed.load_state_dict({'weight' : torch.rand(self.model.embed.weight.size(),device='cuda')})
        self.model.embed.weight.data.zero_()
        self.model.embed.weight.requires_grad = True
        self.model.set_embed_factor(1.0)

class TrainOnlyEmbeds(TrainFullModel):
    def __init__(self, model: SampleEmbeddingNet, optimizer, criterion, epochs, train_loader, test_loader, writer,
                 device, plot_idx_mng: PlotIdxsMng, scheduler):
        super().__init__(model, optimizer, criterion, epochs, train_loader, test_loader, writer, device, plot_idx_mng,
                         scheduler)

    def on_cycle_begin(self):
        for param in self.model.main_net.parameters():
            param.requires_grad = False
        for param in self.model.embed.parameters():
            param.requires_grad = True


class TrainOnlyMainNet(TrainFullModel):

    def __init__(self, model: SampleEmbeddingNet, optimizer, criterion, epochs, train_loader, test_loader, writer,
                 device, plot_idx_mng: PlotIdxsMng, scheduler):
        super().__init__(model, optimizer, criterion, epochs, train_loader, test_loader, writer, device, plot_idx_mng,
                         scheduler)

    def on_cycle_begin(self):
        for param in self.model.main_net.parameters():
            param.requires_grad = True
        for param in self.model.embed.parameters():
            param.requires_grad = False


class TrainOnlyMainNetCosAnnealingEmbedsFactor(TrainOnlyMainNet):
    def __init__(self, model: SampleEmbeddingNet, optimizer, criterion, epochs, train_loader, test_loader, writer,
                 device, plot_idx_mng: PlotIdxsMng, scheduler, initial_embeds_scale, zero_out_threshold, CYCLE_MUL):
        super().__init__(model, optimizer, criterion, epochs, train_loader, test_loader, writer, device, plot_idx_mng,
                         scheduler)
        self.zero_out_threshold = zero_out_threshold
        self.CYCLE_MUL = CYCLE_MUL
        self.initial_embeds_scale = initial_embeds_scale
        self.cos_annealing_embeds_scale = CyclicCosAnnealingScheduler(0.0, initial_embeds_scale, epochs, CYCLE_MUL)

        self.model.set_embed_factor(initial_embeds_scale)

    def on_epoch_end(self):
        super().on_epoch_end()
        val = self.cos_annealing_embeds_scale.step()
        if val < self.zero_out_threshold:
            val = 0
        self.model.set_embed_factor(val)
        # self.writer.add_scalar("Train Params/Embeds Factor", self.model.get_embed_factor(), self.plot_idx_mng.get_plot_idx("Train Params/Embeds Factor"))

class TrainOnlyMainNetZeroOutEmbedsFactor(TrainOnlyMainNet):
    def __init__(self, model: SampleEmbeddingNet, optimizer, criterion, epochs, train_loader, test_loader, writer,
                 device, plot_idx_mng: PlotIdxsMng, scheduler):
        super().__init__(model, optimizer, criterion, epochs, train_loader, test_loader, writer, device, plot_idx_mng,
                         scheduler)
        self.model.set_embed_factor(0.0)