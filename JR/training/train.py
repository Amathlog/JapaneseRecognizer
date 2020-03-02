import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path


def train(train_loader: DataLoader,
          test_loader: DataLoader,
          nb_epochs: int,
          test_every_n_epochs: int,
          model: nn.Module,
          saving_path: Path):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_loss = -1
    for epoch in range(nb_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        if (epoch + 1) % test_every_n_epochs == 0:
            testing_loss = 0.0
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data

                testing_loss += criterion(model(inputs), labels)

            print('Testing loss: %.3f' % (running_loss / len(test_loader)))
            if best_loss == -1 or testing_loss < best_loss:
                best_loss = testing_loss
                model_name = saving_path / f"katakana_epoch{epoch}.model"
                model.save(str(model_name))
