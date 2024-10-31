import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F


class CoarseSegmentationModel(nn.Module):
    def __init__(self):
        super(CoarseSegmentationModel, self).__init__()
        mobilenet = models.mobilenet_v3_small(weights=None)
        self.features = nn.Sequential(*list(mobilenet.features.children())[:-3])
        self.pointwise_conv = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x):
        x = self.features(x)
        x = self.pointwise_conv(x)
        x = self.sigmoid(x) 
        return x