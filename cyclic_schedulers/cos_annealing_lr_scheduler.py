from cyclic_schedulers.cos_annealing_scheduler import CyclicCosAnnealingScheduler, ReversedCosAnnealingScheduler


class LRScheduler:
    def __init__(self, scheduler, optimizer):
        self.scheduler = scheduler
        self.optimizer = optimizer

    def step(self, val=None):
        lr = self.scheduler.step(val)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class CosLRScheduler:
    def __init__(self, optimizer, min_val, max_val, cycle_length, cycle_mul, min_val_mul=1.0, max_val_mul=1.0):
        self.cos_scheduler = CyclicCosAnnealingScheduler(min_val, max_val, cycle_length, cycle_mul, min_val_mul,
                                                         max_val_mul)
        self.lr_scheduler = LRScheduler(self.cos_scheduler, optimizer)

    def step(self):
        return self.lr_scheduler.step()


class ReversedCosLRScheduler:
    def __init__(self, optimizer, min_val, max_val, cycle_length, cycle_mul, min_val_mul=1.0, max_val_mul=1.0):
        self.cos_scheduler = ReversedCosAnnealingScheduler(min_val, max_val, cycle_length, cycle_mul, min_val_mul,
                                                           max_val_mul)
        self.lr_scheduler = LRScheduler(self.cos_scheduler, optimizer)

    def step(self):
        return self.lr_scheduler.step()


if __name__ == "__main__":
    pass
