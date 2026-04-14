
import torch.nn as nn 

class CNNClassifier(nn.Module):
    def __init__(self, 
                conv_layers=[32, 64, 128, 256],
                linear_layers=[128],
                n_input_channels=3, # RGB = 3 | GRAY = 1
                n_clases=10,
                kernel_size=3):
        super(CNNClassifier, self).__init__()

        L = []
        c = n_input_channels # N, 3, Hin, Win

        for n_l in conv_layers:
            L.append(nn.Conv2d(in_channels=c, out_channels=n_l,
                     kernel_size=kernel_size,
                     stride=1,
                     padding=kernel_size//2)) # N, 32, Hout, Wout
            L.append(nn.BatchNorm2d(num_features=n_l)) # Normalizacion sobre N
            L.append(nn.ReLU())
            L.append(nn.MaxPool2d(kernel_size)) # max[kernel_size](img) -> valor_maximo[kernel_size]
            L.append(nn.Dropout(p=0.1))
            c = n_l
        
        self.network = nn.Sequential(*L)

        L2 = []
        for n_l in linear_layers:
            L2.append(nn.Linear(c, n_l))
            L2.append(nn.ReLU())
            L2.append(nn.Dropout(p=0.3))
            c = n_l 

        L2.append(nn.Linear(c, n_clases))
        self.classifier = nn.Sequential(*L2)

    def forward(self, x):
        x = self.network(x) # [N, 256, H, W]
        # GAP -> Global Average Pooling
        x = x.mean(dim=[2, 3]) # [N, 256] # 256 van a ser el promedio 
        x = self.classifier(x) # [N, 256] -> [N, 10]
        return x

# ResNet (como lo hacian mis antepasados)
import torch.nn.functional as F
class ResNet(nn.Module):
    # Skip-Layers
    class Block(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = nn.Conv2d(n_input, n_output,
                                kernel_size=kernel_size,
                                padding=kernel_size//2,
                                stride=2,
                                bias=False)

            self.c2 = nn.Conv2d(n_output, n_output, 
                                kernel_size=kernel_size,
                                padding=kernel_size//2,
                                bias=False)

            self.c3 = nn.Conv2d(n_output, n_output,
                                kernel_size=kernel_size,
                                padding=kernel_size//2,
                                bias=False)
            
            self.b1 = nn.BatchNorm2d(n_output) # gamma y beta
            self.b2 = nn.BatchNorm2d(n_output) # gamma y beta
            self.b3 = nn.BatchNorm2d(n_output) # gamma y beta

            self.skip = nn.Conv2d(n_input, n_output, kernel_size=1, stride=2)
        
        def forward(self, x):
            # Skip-Layers suministrar una imagen de mas alta resolucion al resultado F(x)
            # relu(F(x) + x)
            return F.relu(
                    self.b3(
                        self.c3(
                            F.relu(
                                self.b2(
                                    self.c2(
                                        F.relu(
                                            self.b1(
                                                    self.c1(x)
                                                   )
                                            )
                                        )
                                        )
                                    )
                                )
                            ) + self.skip(x)
                        ) 

    def __init__(self, layers=[32, 64, 128, 256], linear_layers=[128], n_input_channels=3,
                 n_clases=10, kernel_size=3):
        super().__init__()

        L = []
        c = n_input_channels
        for l in layers:
            L.append(self.Block(c, l, kernel_size=kernel_size, stride=2))
            c = l
        
        self.network = nn.Sequential(*L)

        L2 = []
        for n_l in linear_layers:
            L2.append(nn.Linear(c, n_l))
            L2.append(nn.ReLU())
            L2.append(nn.Dropout(p=0.3))
            c = n_l 

        L2.append(nn.Linear(c, n_clases))
        self.classifier = nn.Sequential(*L2)
    
    def forward(self, x):
        x = self.network(x) # [N, 256, H, W]
        x = x.mean(dim=[2, 3]) # GAP -> [N, 256]
        x = self.classifier(x) # [N, 256] -> [N, 10]
        return x

# ResNet (TransferLearning + Fine-tuning)
from torchvision.models import resnet50, ResNet50_Weights
class ResNet50Transfer(nn.Module):
    def __init__(self, linear_layers=[128], n_clases=10, freeze_base=True):
        super(ResNet50Transfer, self).__init__()

        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.base_model = resnet50(weights=self.weights)

        # Fine Tuning
        if freeze_base is True:
            for param in self.base_model.parameters():
                # Estos parametros no se van a mover
                param.requires_grad = False

        # Reemplazar la ultima capa para hacer fine-tuning
        c = self.base_model.fc.in_features

        L2 = []
        for n_l in linear_layers:
            L2.append(nn.Linear(c, n_l))
            L2.append(nn.ReLU())
            L2.append(nn.Dropout(p=0.3))
            c = n_l 

        L2.append(nn.Linear(c, n_clases))
        self.base_model.fc = nn.Sequential(*L2)
    
    def forward(self, x):
        return self.base_model(x)

def save_model(model: nn.Module, name: str):
    from torch import save
    return save(model.state_dict(), name)

def load_cnn(name: str):
    from torch import load
    r = CNNClassifier()
    r.load_state_dict(load(name, map_location='cpu', weights_only=True))
    return r

def load_resnet(name: str):
    from torch import load
    r = ResNet()
    r.load_state_dict(load(name, map_location='cpu', weights_only=True))
    return r

def load_resnet_transfer(name: str):
    from torch import load
    r = ResNet50Transfer()
    r.load_state_dict(load(name, map_location='cpu', weights_only=True))
    return r