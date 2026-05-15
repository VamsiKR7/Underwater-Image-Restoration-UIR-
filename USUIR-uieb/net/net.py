import torch
from net.ITA import JDSNet, TDNet


class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_net = JDSNet()
        self.mask_net = TDNet()

    def forward(self, data):
        x_j = self.image_net(data)
        x_t = self.mask_net(data)
        return x_j, x_t


