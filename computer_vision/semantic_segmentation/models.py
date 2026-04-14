import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet18, ResNet18_Weights

class UNet(nn.Module):
    class Block(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size//2)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size//2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))
    
    class UpBlock(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size//2, stride=2, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, input_channels=3, layers=[16, 32, 64, 128], n_classes=1, kernel_size=3,
                 use_skip=True):
        
        super().__init__()   
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [input_channels] + layers[:-1]

        c = input_channels
        for i, l in enumerate(layers):
            self.add_module(f"conv{i}", self.Block(c, l, kernel_size=kernel_size, stride=2))
            c = l 
        
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module(f'upconv{i}', self.UpBlock(c, l, kernel_size=kernel_size, stride=2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]

        self.classifier = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x):
        up_activations = []
        for i in range(self.n_conv):
            up_activations.append(x)
            x = self._modules[f"conv{i}"](x)
        
        for i in reversed(range(self.n_conv)):
            x = self._modules[f"upconv{i}"](x)

            # Arreglar dimensiones 
            x = x[:, :, :up_activations[i].size(2), :up_activations[i].size(3)]

            if self.use_skip:
                x = torch.cat([x, up_activations[i]], dim=1)
            
        return self.classifier(x)   
    
class TransferSegmentation(nn.Module):
    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=2,
                      stride=2):
            super().__init__()
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

            self.conv = nn.Sequential(
                nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
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

        # Decoder
        '''self.up4 = self.UpBlock(512, 256)
        self.up3 = self.UpBlock(256, 128)
        self.up2 = self.UpBlock(128, 64)
        self.up1 = self.UpBlock(64, 64)'''
        auxImg = torch.randn(1, 3, 256, 256)  # Imagen auxiliar para ver las dimensiones
        with torch.no_grad():
            encoder_shapes, _ = self.get_encoder_shapes(auxImg) 

        self.up_blocks = nn.ModuleList()
        for i in range(len(encoder_shapes) - 1):
            in_ch = encoder_shapes[i][1]
            skip_ch = encoder_shapes[i+1][1]
            out_ch = skip_ch
            self.up_blocks.append(self.UpBlock(in_ch, out_ch))

        self.final_conv = nn.Conv2d(out_ch, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.in_conv(x)
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # Decoder skip connections
        z = self.up_blocks[0](x5, x4)
        z = self.up_blocks[1](z, x3)
        z = self.up_blocks[2](z, x2)
        z = self.up_blocks[3](z, x0)

        out = self.final_conv(z)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

    def get_encoder_shapes(self, dummy_input):
        # Paso la imagen generada por las convoluciones del encoder para ver las dimensiones en cada una
        # asi puedo empatar las dimensiones en el upblock y aplicar skip connections
        shapes = []
        x = self.in_conv(dummy_input) 
        x0 = x
        shapes.append(x.shape)

        x = self.maxpool(x)
        x1 = self.layer1(x)
        shapes.append(x1.shape)

        x2 = self.layer2(x1)
        shapes.append(x2.shape)

        x3 = self.layer3(x2)
        shapes.append(x3.shape)

        x4 = self.layer4(x3)
        shapes.append(x4.shape)
        return shapes[::-1], [x4, x3, x2, x1, x0]  # La mas profunda primero

def save_model(model, name):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))

def load_model(input_channels=1, name='model.th'):
    from torch import load
    from os import path
    try:
        r = UNet(input_channels=input_channels)
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    except:
        r = TransferSegmentation(input_channels=input_channels)
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    
    return r
