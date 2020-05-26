import torch.nn as nn
from torch.autograd import Function

''' 
Very easy template to start for developing your AlexNet with DANN 
Has not been tested, might contain incompatibilities with most recent versions of PyTorch (you should address this)
However, the logic is consistent
'''

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class RandomNetworkWithReverseGrad(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(RandomNetworkWithReverseGrad, self).__init__()
        #features extractor:
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        #class classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

        #domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100), 
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 1000),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, alpha=None):
        features = self.features
        # Flatten the features:
        features = features.view(features.size(0), -1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(features, alpha)
            discriminator_output = self.domain_classifier(features)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            # do something else
            class_outputs = self.classifier(features)
            return class_outputs

def alexNetDA(pretrained=True,  num_classes=7, **kwargs):
    net = RandomNetworkWithReverseGrad(num_classes = 7, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'])
        net.load_state_dict(state_dict, strict=False)
        for x in [(0,1) , (3,6)]:
            net.domain_classifier[x[0]].weight.data = net.classifier[x[1]].weight.data
            net.domain_classifier[x[0]].bias.data = net.classifier[x[1]].bias.data
        net.classifier[6] = nn.Linear(4096, num_classes)
        net.domain_classifier[3] = nn.Linear(100, 2)
    return net     
