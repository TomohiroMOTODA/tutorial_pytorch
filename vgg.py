import torch.nn as nn
from torchvision import models, transforms

# TODO: 動作確認
class vgg16(nn.model):
    def __init__(self):
        super(vgg16, self).__init__()
        self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(224, 224, 64)
        self.conv1_2 = nn.Conv2d(224, 224, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_1 = nn.Conv2d(112, 112, 128)
        self.conv2_2 = nn.Conv2d(112, 112, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_1 = nn.Conv2d(56, 56, 256)
        self.conv3_2 = nn.Conv2d(56, 56, 256)
        self.conv3_3 = nn.Conv2d(56, 56, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv4_1 = nn.Conv2d(28, 28, 512)
        self.conv4_2 = nn.Conv2d(28, 28, 512)
        self.conv4_3 = nn.Conv2d(28, 28, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5_1 = nn.Conv2d(14, 14, 512)
        self.conv5_2 = nn.Conv2d(14, 14, 512)
        self.conv5_3 = nn.Conv2d(14, 14, 512)
        self.pool5 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(7 * 7 * 512, 4096) # 全結合層, input_size, output_size
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.softmax = nn.Softmax(1)

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x