import torch
import torch.nn as nn
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# for word 'hello' -> 'ohlol'
input_size = 4
hidden_size = 4
batch_size = 1
num_layers = 1
seq_len = 5

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


# define a rnn class
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

    def forward(self, x):
        h0 = torch.zeros(num_layers, batch_size, hidden_size)
        out, _ = self.rnn(x, h0)
        out = out.view(-1, hidden_size)
        return out


model = RNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# train model
for epoch in range(15):
    # training step
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    idx = torch.max(outputs, dim=1)[1]
    idx = idx.data.numpy()
    print('Predicted string: ', "".join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
