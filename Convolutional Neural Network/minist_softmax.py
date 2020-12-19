import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# define batch size
batch_size = 100
# transform to set up image data, the dim needs c x h x w
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standardize dataset
])

# define training and testing data
train_dataset = datasets.MNIST(root="C:/Users/liz27/Documents/deep learning\deep neural network/softmax with minist",
                               train=True,
                               download=False,
                               transform=transform)
test_dataset = datasets.MNIST(root="C:/Users/liz27/Documents/deep learning\deep neural network/softmax with minist",
                              train=False,
                              download=False,
                              transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


# define a neural net classifier
class NeuralNet(nn.Module):
    def __init__(self, d, h1, h2, h3, h4, n):
        super(NeuralNet, self).__init__()
        layers = [
            nn.Linear(d, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, h4),
            nn.ReLU(),
            nn.Linear(h4, n)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layers(x)
        return x


# define model object
model = NeuralNet(784, 512, 256, 128, 64, 10)

# define loss function and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda')
model.to(device)


# define a training function
def train(epo):
    running_loss = 0.0
    for batch_id, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        inputs, target = data
        # put into GPU
        inputs = inputs.to(device)
        target = target.to(device)

        # forward -> compute cost -> backward -> update parameters
        outputs = model(inputs)
        cost = loss(outputs, target)
        cost.backward()
        optimizer.step()

        # print the training process per batch
        running_loss += cost.cpu().item()
        if (batch_id + 1) % 300 == 0:
            print("epoch: {}, batch id: {}, loss: {}".format(epo + 1, batch_id + 1, running_loss / 300))
            running_loss = 0.0  # reset loss into 0 for next batch


# define a testing function
def test():
    correct = 0
    total = 0
    # no grad needs in this time
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # put into GPU
            images = images.to(device)
            labels = labels.to(device)

            # predict
            outputs = model(images)
            outputs = outputs.cpu()
            _, prediction = torch.max(outputs.data, dim=1)

            # compute accuracy
            total += labels.size(0)
            correct += (prediction == labels.cpu()).sum().item()
            accuracy = correct / total
        print('Accuracy on test set: {}'.format(accuracy))


# test code
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
