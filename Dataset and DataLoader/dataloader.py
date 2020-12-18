import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# define a dataset class
class CreateDatset(Dataset):
    def __init__(self, file):
        xy = np.loadtxt(file, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# define a classifier class
class NeuratNet(nn.Module):
    def __init__(self, d, h1, h2, h3, m):
        super(NeuratNet, self).__init__()
        layers = [
            nn.Linear(d, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, m),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# training model
if __name__ == '__main__':
    # define dataset object
    dataset = CreateDatset('diabetes.csv')

    # apply data loader
    train_loader = DataLoader(dataset=dataset, batch_size=10)

    # define the neural network
    model = NeuratNet(8, 10, 10, 8, 1)

    # define optimizer and loss
    loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # training 
    epochs = 100
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            # get data
            X, y = data
            # forward propagation
            y_pred = model(X)
            # compute cost
            cost = loss(y_pred, y)
            # backward propagation
            optimizer.zero_grad()
            cost.backward()
            # updating weights
            optimizer.step()

            # print training process
            print('epoch: {}, iteration: {}, cost: {}'.format(epoch, i, cost))
