import torch
import torch.nn as nn
import torch.functional as F

class SampleEmbeddingNet(nn.Module):
    def __init__(self, main_net: nn.Module, embeddings_count, embed_dim, embed_transform: nn.Module, embed_factor, embed_max_norm):
        super().__init__()
        self.main_net = main_net
        self.embed_transform = embed_transform

        self.embed_max_norm = embed_max_norm
        self.embed_dim = embed_dim
        self.embed_factor = embed_factor

        self.embeddings_count = embeddings_count

        self.embed = nn.Embedding(self.embeddings_count, self.embed_dim, max_norm=self.embed_max_norm)

    def set_embed_factor(self, val):
        self.embed_factor = val

    def get_embed_factor(self):
        return self.embed_factor

    def forward(self, x):
        if not self.training:
            return self.main_net(x)
        else:
            x, idxs = x
            embeds = self.embed(idxs)
            embeds = self.embed_transform(embeds)
            x = x + embeds
            return self.main_net(x)

    def get_embeds_norm(self):
        return self.embed.weight.norm()

    def get_main_net_norm(self):
        norm = 0.0
        for param in self.main_net.parameters():
            norm += param.norm()
        return norm

class EmbedUpsample(nn.Module):
    def __init__(self, upsample_factor):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.upsample = nn.Upsample(scale_factor=self.upsample_factor, mode='bilinear')
    def forward(self, x):
        bs = x.size(0)
        img_size = int(32 / self.upsample_factor)
        x = x.view(bs, 3, img_size,img_size)
        return self.upsample(x)

class NoUpsample(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.size(0), 3, 32, 32)
