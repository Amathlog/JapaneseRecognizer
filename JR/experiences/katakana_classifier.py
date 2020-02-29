from torch.utils.data import DataLoader

from JR.data.katakana import Katakana, KatakanaTestingDataset, KatakanaTrainingDataset, NB_CLASSES, IMG_DIM
from JR.training.mlp_model import MLPClassifier
from JR.training.cnn_model import CNNClassifier
from JR.training.train import train as NNTrain
from JR.utils import get_nb_threads
from JR.data.config import DATA_DIR

import matplotlib.pyplot as plt

# First load data
full_dataset = Katakana()
training_dataset = KatakanaTrainingDataset(full_dataset)
testing_dataset = KatakanaTestingDataset(full_dataset)

# Then create the dataloader associated
batch_size = 32
num_workers = get_nb_threads()
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Create the model
model = CNNClassifier(IMG_DIM, NB_CLASSES)


def train():
    # Then train!
    print("Start training")
    NNTrain(training_dataloader, testing_dataloader, 100, 5, model, DATA_DIR)
    print("Training is over")

def test(model_to_load):
    model.load(str(model_to_load))
    for i, data in enumerate(testing_dataloader, 0):
        inputs, labels = data

        predicted = model(inputs).argmax(dim=-1).numpy()

        for input, label, predicted_label in zip(inputs, labels, predicted):
            label_class = full_dataset.classes[label.numpy()]
            predicted_class = full_dataset.classes[predicted_label]

            if label_class == predicted_class:
                continue

            plt.imshow(input.numpy().reshape(64,64))
            plt.title(f"Class: {label_class} ; Predicted: {predicted_class}")
            plt.show()
            plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        if sys.argv[1] == "--test":
            test(sys.argv[2])
            exit(0)
    train()
