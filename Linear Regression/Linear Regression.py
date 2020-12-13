#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# design feature X and label y
x_values = [i for i in range(11)]
x_train = np.array(x_values,dtype=np.float32)
x_train = x_train.reshape(-1,1)
x_train.shape


# In[3]:


y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values,dtype=np.float32)
y_train = y_train.reshape(-1,1)


# In[4]:


# linear regression
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out
    
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)


# In[5]:


# define optimizer and loss
epochs = 1000
eta = 0.01
optimizer = torch.optim.SGD(model.parameters(), eta)
loss = nn.MSELoss()
costs = []
# training process
for epoch in range(epochs):
    epoch += 1
    
    # conver training data into tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    
    # clean gradient
    optimizer.zero_grad()
    
    # forward
    outputs = model.forward(inputs)
    
    # compute cost
    cost = loss(outputs,labels)
    costs.append(cost.detach().numpy())
    # backward
    cost.backward()
    
    # updating weights
    optimizer.step()
    
    # print training process
    if epoch % 50 == 0:
        print('epoch: {}, cost: {}'.format(epoch,cost))   


# In[6]:


# predict
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
predicted


# In[7]:


torch.save(model.state_dict(),'model.pkl')


# In[8]:


model.load_state_dict(torch.load('model.pkl'))

