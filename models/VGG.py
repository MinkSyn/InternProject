import torch
import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        #Inheritance from nn.Module
        super(VGG, self).__init__()
        
        self.features = features
        #Layer Fully Connected
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self.initialize_weights()
    
    def forward_once(self, x):
        x = self.features(x)
        #Layer Flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3
     
    def initialize_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                           
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
         
    
#Layer Convolution
def make_layers(cfg, batch_norm=False):
    all_layers = []
    input_channels = 3 #Color images have there color
    
    for layer in cfg:
        #Create Layer Pooling
        if layer == 'M':
            all_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #Create Layer Conv + BatchNorm + ReLU
        else:
            conv2d = nn.Conv2d(input_channels, layer, kernel_size=3, padding=1)
            if batch_norm:
                all_layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
            else:
                all_layers += [conv2d, nn.ReLU(inplace=True)]
            input_channels = layer
    return nn.Sequential(*all_layers)