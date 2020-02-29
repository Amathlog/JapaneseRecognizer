import torch.nn as nn
import torch.nn.functional as F

from JR.training.model import Classifier


class CNNClassifier(Classifier):
    def __init__(self, input_shape, nb_classes, hidden_layers=(128, 128)):
        super(CNNClassifier, self).__init__(input_shape, nb_classes)

        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.input_size = 6 * 13 * 13

        sizes = (self.input_size,) + hidden_layers + (nb_classes,)

        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])

    def forward(self, x):
        x = x.view(-1, 1, *self.input_shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.input_size)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
