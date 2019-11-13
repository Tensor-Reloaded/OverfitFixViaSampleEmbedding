from collections import defaultdict


class PlotIdxsMng:
    def __init__(self):
        self.dict = defaultdict(lambda : 0)

    def get_plot_idx(self, tag):
        value = self.dict[tag]
        self.dict[tag] += 1
        return value
