import torch.nn as nn
import torch.nn.functional as F

from JR.training.model import Classifier


class MLPClassifier(Classifier):
    def __init__(self, input_shape, nb_classes, hidden_layers=(1024, 512, 256, 128)):
        super(MLPClassifier, self).__init__(input_shape, nb_classes)

        sizes = (self.input_size,) + hidden_layers + (nb_classes,)

        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)])

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
