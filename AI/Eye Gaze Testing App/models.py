import torch
import torchvision
from torchvision import models
from ultralytics import YOLO
from torch import nn

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=8, input_features=576, hidden_features=1024, output_features=1280):
        super(MobileNetV3Small, self).__init__()
        self.model = torchvision.models.mobilenet_v3_small(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_features, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(hidden_features, output_features, bias=True),
            nn.Linear(output_features, num_classes, bias=True),  # Output layer with 8 classes
            nn.Softmax(dim=1)
        )
        self.model.classifier = self.classifier

    def load_weights(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    def forward(self, x):
        x = self.model(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def load_weights(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    def forward(self, x):
        x = self.model(x)
        return x

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileNetV2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True),
            nn.Softmax(dim=1)
        )

    def load_weights(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    def forward(self, x):
        x = self.model(x)
        return x

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=8):
        super(SqueezeNet, self).__init__()
        self.model =torchvision.models.squeezenet1_0(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 8, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def load_weights(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    def forward(self, x):
        x = self.model(x)
        return x


# model paths
model_path_MNV3 = "MNV3_small_weights_cleaned_100.pth"
#model_path = "eyegaze.pt"
#model_path_MNV2 = "MNV2_weights.h5"
model_path_SqueezeNet = "Squeezenet.pt"
model_path_yolov8 = "yolov8n-face.pt"

# instances of each model
# model_MNV3 = MobileNetV3Small()
# model_MNV3.load_weights(model_path_MNV3)

# model_ResNet50 = ResNet50()
# model_ResNet50.load_weights(model_path)

# model_MNV2 = MobileNetV2()
# model_MNV2.load_weights(model_path_MNV2)

model_SqueezeNet = SqueezeNet()
model_SqueezeNet.load_weights(model_path_SqueezeNet)

model_yolo = YOLO(model_path_yolov8)
