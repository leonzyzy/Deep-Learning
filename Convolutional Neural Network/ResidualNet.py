import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# transform to set up image data, the dim needs c x h x w
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standardize dataset
])

# define training and testing set with mnist
train_set = datasets.MNIST(
    root="C:/Users\liz27/Documents/deep learning/convolutional neural network",
    train=True,
    download=False,
    transform=transform
)
test_set = datasets.MNIST(
    root="C:/Users\liz27/Documents/deep learning/convolutional neural network",
    train=False,
    download=False,
    transform=transform
)

# define training and testing loader
batch_size = 64
train_loader = DataLoader(
    train_set,
    shuffle=True,
    batch_size=batch_size
)
test_loader = DataLoader(
    test_set,
    shuffle=True,
    batch_size=batch_size
)


# define a residual net block
class ResidualNet(nn.Module):
    def __init__(self, c):
        super(ResidualNet, self).__init__()
        self.Conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x):
        a = F.relu(self.Conv1(x))
        a = self.Conv2(x)
        return F.relu(a + x)


# define a CNN class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualNet(16)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualNet(32)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# define loss function and optimizer
model = ConvNet()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# put model into gpu
device = torch.device('cuda')
model.to(device)


# define a train function
def train(epo):
    train_loss = 0.0
    for batch_id, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # put train data into gpu

        # forward -> compute cost -> backward -> updating parameters
        output = model(inputs)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        # print the training loss
        train_loss += loss.item()
        if (batch_id + 1) % 100 == 0:
            print("epoch: {}, batch id: {}, training loss: {:.6f}".format(epo + 1, batch_id + 1, train_loss / 100))
            train_loss = 0.0


# define a test function
def test():
    correct = 0
    total_size = 0
    # no grad needs in testing
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # put test data into gpu

            # predict
            outputs = model(images)
            predictions = torch.max(outputs, dim=1)[1]

            # compute accuracy
            total_size += labels.size(0)
            correct += (predictions == labels).sum().item()
    print("Accuracy on testing set: {:.3f}".format(correct / total_size))


# run code
if __name__ == '__main__':
    for epoch in range(5):
        train(epoch)
        test()
