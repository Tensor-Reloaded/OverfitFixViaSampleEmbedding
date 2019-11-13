from torch.utils.data import Dataset

class SampleIdxDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        data, target = self.ds[index]
        return (data, index), target

    def __len__(self):
        return len(self.ds)