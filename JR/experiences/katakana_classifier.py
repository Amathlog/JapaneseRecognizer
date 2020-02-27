from torch.utils.data import DataLoader

from JR.data.katakana import Katakana, KatakanaTestingDataset, KatakanaTrainingDataset, NB_CLASSES, IMG_DIM
from JR.training.mlp_model import MLPClassifier
from JR.training.train import train
from JR.utils import get_nb_threads

# First load data
full_dataset = Katakana()
training_dataset = KatakanaTrainingDataset(full_dataset)
testing_dataset = KatakanaTestingDataset(full_dataset)

# Then create the dataloader associated
num_workers = get_nb_threads()
training_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers=num_workers)
testing_dataloader = DataLoader(testing_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

# Create the model
model = MLPClassifier(IMG_DIM, NB_CLASSES)

# Then train!
print("Start training")
train(training_dataloader, testing_dataloader, 2, 2, model)
print("Training is over")
