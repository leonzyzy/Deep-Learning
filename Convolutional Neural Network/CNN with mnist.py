import torch
import torch.nn as nn
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


# define a convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.Conv1 = nn.Sequential(  # input size: 1 x 28 x 28
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 16 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2)  # output size: 16 x 14 x 14
        )
        self.Conv2 = nn.Sequential(  # input size: # 16 x 14 x 14
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 32 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2)  # output size: 32 x 7 x 7
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
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
            print("epoch: {}, batch id: {}, training loss: {:.6f}".format(epo, batch_id+1, train_loss / 100))
            train_loss = 0.0


# define a test function
def test():
    correct = 0
    total_size = 0
    # no grad needs in testing
    with torch.no_grad():
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
