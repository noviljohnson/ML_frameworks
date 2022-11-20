import torch

# x = torch.empty(1)
# print(x)
# x = torch.empty(1,2)
# print(x)
# x = torch.empty(1,2,3)
# print(x)
# x = torch.empty(1,2,3,4)
# print(x)

# x = torch.rand(2,2)
# print(x)
# x = torch.zeros(2,2) 
# print(x)
# x = torch.ones(2,2)
# print(x)

# x = torch.ones(2,2, dtype=torch.int)
# print(x)
# print(x.dtype)
# print(x.size())

# x = torch.tensor([2.3,5,6])
# print(x,x.size(),x.dtype)

# x = torch.rand(2,2)
# y = torch.rand(2,2)
# print(x,y)
# z = x+y
# print(z)
# z = torch.add(x,y)
# print(z)

# y.add_(x)   # _ inplace operation
# print(y)
# y.sub_(x)
# print(y)
# y.mul_(x)
# print(y)
# y.div_(x)
# print(y)

# y = torch.rand(4,4)
# print(y.size())
# x = y.view(16)
# print(x.size())

import numpy as np

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# # if tensor is on cpu not on gpu then both a,b share(points to) same memory location 
# # if we modify a, b will also be modified 

# a.add_(1)
# print(a)
# print(b)

# x = np.ones(5)
# print(x,'x')
# y = torch.from_numpy(x)#, dtype=)
# print(y,'y')

# x += 1
# print(x,'x')
# print(y,'y')

# if we have gpu / cuda

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x + y
#     z = z.to("cpu")

# x = torch.ones(5, requires_grad=True)  #need calculates gradients for this tensor 
# # when ever there is var that needs to be optimized  we need gradients
# # then we need to specify requires_grad=True
# print(x)

## Autograd pkg 
## gradients are essential for model optimization

# x = torch.randn(3, requires_grad=True)
# print(x)

# y = x+2   # creates a computational graph
#           x ->
#                ( + ) -> y
#           2 -> 
# print(y)

# z = y*y*2
# # z = z.mean()
# print(z )

# # scalars 
# v = torch.tensor([0.1,2,4], dtype=torch.float) # if z = z.mean() commented  then add v as arg in z.backward()
# print(z.backward(v))  # dz/dx
# print(z)
# print(x.grad)        # vector jabcobian product


# ways to prevent pytorch to track gradients/history
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

# training example

# weights = torch.ones(4, requires_grad=True)

# for epoch in range(2):
#     model_output = (weights*3).sum()

#     model_output.backward()  # previous gradients will be summed up 

#     print(weights.grad)
    
#     weights.grad.zero_()     # we must empty the gradients 


# pytorch builtin optimizers
# optimizer 

# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()


## Backpropagation algorithm
## - chain rule for derivatives
## - computational graph, local gradients
## -- steps
## 1. Forward pass - computation loss
## 2. Compute local gradients
## 3. backward pass: compute dLoss / dWeights using chain rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w*x
loss = (y_hat - y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)

## update weights
## nxt iterations : forward and backward props
# -----------

## optimize model  with automatic gradient computation
## init : steps  
## prediction : manually
## gradients computation : manually
## loss computation : manually
## parameter updates : manually 

## Goal :  steps -> compute all the above steps using torch pkgs
## prediction : pytorch model
## gradients computation : autograd
## loss computation : pytorch loss
## parameter updates : pytrch optimizer 


## init : steps

import numpy as np
# f = w * x
# f = 2 * x

x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

# weights 0 init
w = 0.0

# model prediction
def forward(x):
    return w*x

# loss = mse
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

# gradient
# mse = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x(w*x - y)

def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before training : f(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
n_iters = 20#10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)

    # loss
    l = loss(y,y_pred)

    # gradients
    dw = gradient(x,y,y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 1 ==0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training : f(5) = {forward(5):.3f}')

