# 1. Design model (input, outpu size, forward pass)
# 2. Construct loss and optimizer
# 3. training loop
#   - forward pass: compute prediction and loss
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0. prepare data
x_nmpy, y_nmpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
# convert to a torch tensor
x = torch.from_numpy(x_nmpy.astype(np.float32))
y = torch.from_numpy(y_nmpy.astype(np.float32))
y = y.view(y.shape[0], 1)   # reshape to col vector

n_samples, n_features = x.shape
print(n_samples, n_features,)

# 1. model
# one layer - use builtin linear model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2. loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3. training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(x)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad() # empty the gradients

    if (epoch +1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
# detach the tensor before converting into numpy
predicted = model(x).detach().numpy()
plt.plot(x_nmpy, y_nmpy, 'ro')
plt.plot(x_nmpy, predicted, 'b')
plt.show()
#1:39 