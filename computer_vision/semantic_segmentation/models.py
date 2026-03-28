from torchvision.models import resnet18, ResNet18_Weights
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class TransferSegmentation(nn.Module):
    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
            super().__init__()
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

            self.conv = nn.Sequential(
                nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        def forward(self, x, skip):
            x = self.upconv(x)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)
        
    def __init__(self, n_classes=1):
        super().__init__()
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.encoder = resnet18(weights=self.weights)

        self.in_conv = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu
        )

        self.maxpool = self.encoder.maxpool 
        self.layer1 = self.encoder.layer1 
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        self.up4 = self.UpBlock(512, 256)
        self.up3 = self.UpBlock(256, 128)
        self.up2 = self.UpBlock(128, 64)
        self.up1 = self.UpBlock(64, 64)

        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoders (La U de bajada)
        x0 = self.in_conv(x)
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # Decoders (La U de subida)
        z = self.up1(x5, x4)
        z = self.up2(z, x3)
        z = self.up3(z, x2)
        z = self.up4(z, x1)

        out = self.final_conv(z)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out