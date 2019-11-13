import math


class CyclicCosAnnealingScheduler(object):
    def __init__(self, min_val, max_val, cycle_length, cycle_mul, min_val_mul=1.0, max_val_mul=1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.cycle_length = cycle_length
        self.cycle_mul = cycle_mul
        self.min_val_mul = min_val_mul
        self.max_val_mul = max_val_mul

        self.cur_iteration = 0

    def get_value(self):
        width = self.cycle_length
        curr_pos = self.cur_iteration
        return self.min_val + (self.max_val - self.min_val) * (1 + math.cos(math.pi * curr_pos / width)) / 2

    def step(self, val=None):
        if (self.cur_iteration + 1) % self.cycle_length == 0:
            self.cycle_length = max(1, int(self.cycle_length * self.cycle_mul))
            self.min_val *= self.min_val_mul
            self.max_val *= self.max_val_mul
            self.cur_iteration = -1
        self.cur_iteration += 1
        return self.get_value()


class ReversedCosAnnealingScheduler(CyclicCosAnnealingScheduler):
    def __init__(self, min_val, max_val, cycle_length, cycle_mul, min_val_mul, max_val_mul):
        super().__init__(min_val, max_val, cycle_length, cycle_mul, min_val_mul, max_val_mul)

    def step(self, val=None):
        return self.max_val - super().step()
