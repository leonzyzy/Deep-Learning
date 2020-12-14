import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# --- The goal is to predict the credit limit
# import data
def dataImport(file):
    data = pd.read_csv(file)
    return data

# preprocessing data
def feature_processing(df):
    # get label
    label = np.array(df['Credit_Limit'])
    # set features
    features = df[['Attrition_Flag', 'Customer_Age', 'Gender',
       'Dependent_count', 'Education_Level', 'Marital_Status',
       'Income_Category', 'Card_Category', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon','Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]
    # one hot encoding for categorical variables
    features = pd.get_dummies(features)
    # standardize features
    features = np.array(preprocessing.StandardScaler().fit_transform(features))
    return features,label

# training testing split
bank_data = dataImport('BankChurners.csv')
X,y = feature_processing(bank_data)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123)

# set training/testing data into tensor format
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).reshape(-1,1).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).reshape(-1,1).float()

# define neural network class
class NeuralNet(nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_output):
        super(NeuralNet, self).__init__()
        self.h1 = nn.Linear(n_input,n_hidden1)
        self.h2 = nn.Linear(n_hidden1,n_hidden2)
        self.output = nn.Linear(n_hidden2,n_output)

    def forward(self,x):
        x = F.relu_(self.h1(x))
        x = F.relu_(self.h2(x))
        out = self.output(x)
        return out

# create neural net object
net = NeuralNet(38,38,38,1)

# optimizer and loss function
epochs = 10000 # training times
eta = 0.01 # learning rate
optimizer = torch.optim.Adam(net.parameters(),lr=eta)
loss = torch.nn.MSELoss(reduction='mean') # mean square loss
costs = []

# training
for i in range(epochs):
    optimizer.zero_grad()

    # forward
    pred = net(X_train_tensor)

    # compute cost
    cost = loss(pred,y_train_tensor)

    # backward
    cost.backward()
    optimizer.step()

    # show training process
    if i % 100 == 0:
        costs.append(cost.data.numpy())
        print("epoch: {}, cost: {}".format(i,cost))

# plt.plot(costs)
# plt.show()

# prediction
y_pred = net(X_train_tensor).detach().numpy()
print(np.mean(y_pred == y_test.reshape(-1,1)))