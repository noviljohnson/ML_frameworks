# Activation functions
# apply a non-linear transformation and decide whether a neuron should be activated or not.
# after each layer we use an activation funtion
# most popular activation functions

# step function
# sigmoid
# TanH
# ReLU
# Leaky ReLU
# Softmax

# sigmoid
"""
f(x) = 1 / ( 1 + e**(-x))

-> 0 <= f(x) <= 1
-> typically in the last layer of a binary classification problem
"""

# TanH
"""
f(x) = ( 2 / (1 + e**(-2*x)) ) - 1

-> -1 <= f(x) <= 1
-> hidden layers
"""

# ReLU
"""
f(x) = max(0,x)

-> if you don't know what to use, just use a ReLU for hidden layers
"""

# 