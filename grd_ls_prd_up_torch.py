## goal: steps

## Goal :  steps -> compute all the above steps using torch pkgs
## prediction : pytorch model
## gradients computation : autograd
## loss computation : pytorch loss
## parameter updates : pytrch optimizer 

## 1. Design model (input, output size, forward pass)
## 2. construct loss and optimizer
## 3. Training loop
## - forward pass: compute prediction
## - backward pass: gradients
## - update weights

import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)

n_samples,n_features = x.shape
input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)

# custome model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)    

model = LinearRegression(input_size, output_size)

# weights 0 init
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# def forward(x):
#     return w * x
## repalce manually implemented forward pass

print(f'Prediction before training : f(5) = {model(x_test).item():.3f}') #{forward(5):.3f}')

# training 
learning_rate = 0.02
n_iters = 30#20#10

# loss from nn
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(x)

    # loss
    l = loss(y,y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 1 ==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')


print(f'Prediction after training : f(5) = {model(x_test).item():.3f}') #{forward(5):.3f}')
