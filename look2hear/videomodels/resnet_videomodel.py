"""
ResNet-based Video Model for extracting visual features from lip images
"""
import torch
import torch.nn as nn
import math
import numpy as np


def threeD_to_2D_tensor(x):
    """Convert 3D tensor to 2D for ResNet processing"""
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class BasicBlock(nn.Module):
    """Basic ResNet Block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='relu'):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu', 'prelu']

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    """ResNet for video feature extraction"""

    def __init__(self, block, layers, num_classes=512, relu_type='relu'):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.relu_type = relu_type

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNetVideoModel(nn.Module):
    """
    ResNet-based Video Model for lip reading feature extraction

    Args:
        relu_type: Type of ReLU activation ('relu' or 'prelu')
        pretrain: Path to pretrained weights
    """

    def __init__(self, relu_type="prelu", pretrain=None):
        super(ResNetVideoModel, self).__init__()

        self.frontend_nout = 64
        self.backend_out = 512

        # ResNet trunk
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        # Frontend 3D convolution
        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout)
            if relu_type == "prelu"
            else nn.ReLU()
        )

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Initialize weights
        self._initialize_weights_randomly()

        # Load pretrained weights if provided
        if pretrain:
            self.init_from(pretrain)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] video frames

        Returns:
            out: [B, D, T] video features
        """
        B, C, T, H, W = x.size()

        # Frontend 3D convolution
        x = self.frontend3D(x)
        Tnew = x.shape[2]

        # Convert to 2D for ResNet
        x = threeD_to_2D_tensor(x)

        # ResNet feature extraction
        x = self.trunk(x)

        # Reshape to [B, Tnew, D]
        x = x.view(B, Tnew, x.size(1))

        # Transpose to [B, D, T]
        return x.transpose(1, 2)

    def _initialize_weights_randomly(self):
        """Initialize weights"""
        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt(2.0 / float(n))
        else:
            def f(n):
                return 2.0 / float(n)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

    def init_from(self, path):
        """Load pretrained weights"""
        print(f"Loading video model from: {path}")
        pretrained_dict = torch.load(path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["model_state_dict"]

        # Update model
        model_dict = self.state_dict()
        update_dict = {}

        for k, v in pretrained_dict.items():
            # Skip TCN layers if present
            if "tcn" not in k and k in model_dict:
                update_dict[k] = v

        if len(update_dict) > 0:
            print(f"Loaded {len(update_dict)} layers from pretrained model")
            model_dict.update(update_dict)
            self.load_state_dict(model_dict)

            # Freeze pretrained weights
            for p in self.parameters():
                p.requires_grad = False
        else:
            print("Warning: No matching layers found in pretrained model!")

        return self
