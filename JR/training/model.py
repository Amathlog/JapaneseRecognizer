import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(Classifier, self).__init__()

        self.input_size = input_shape[0] * input_shape[1]
        self.nb_classes = nb_classes

    def forward(self, x):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
