
# import numpy as np
import torch
# f = w * x
# f = 2 * x

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

# weights 0 init
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w*x

# loss = mse
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

print(f'Prediction before training : f(5) = {forward(5):.3f}')

# training
learning_rate = 0.02
n_iters = 30#20#10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)

    # loss
    l = loss(y,y_pred)

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()
    if epoch % 1 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training : f(5) = {forward(5):.3f}')

