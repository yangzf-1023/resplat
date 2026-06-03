import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50


def _build_resnet(resnet_layers):
    """Build a torchvision ResNet backbone without implicit network access.

    ReSplat loads its own checkpoint immediately after model construction.  The
    original code requested torchvision's ImageNet-pretrained ResNet weights at
    construction time, which can trigger a download in fresh/offline experiment
    environments and fail before the ReSplat checkpoint is loaded.  Constructing
    the backbone with random weights is enough here because the following
    checkpoint load overwrites the model parameters.
    """
    builders = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
    }
    if resnet_layers not in builders:
        raise NotImplementedError

    builder = builders[resnet_layers]
    try:
        return builder(weights=None)
    except TypeError:
        # Compatibility with older torchvision releases.
        return builder(pretrained=False)


class ResNetFeatureWarpper(nn.Module):
    def __init__(self, shallow_resnet_feature=False,
                 resnet_layers=18,
                 ):
        super(ResNetFeatureWarpper, self).__init__()

        self.shallow_resnet_feature = shallow_resnet_feature

        resnet = _build_resnet(resnet_layers)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        if not shallow_resnet_feature:
            self.layer2 = resnet.layer2

    def forward(self, x):
        out = []
        x = self.conv1(x)
        out.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out.append(x)

        if not self.shallow_resnet_feature:
            x = self.layer2(x)
            out.append(x)

        return out
