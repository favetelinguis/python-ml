# Example from https://roshansanthosh.wordpress.com/2015/02/22/linear-regression-in-theano/
# Objective: Regression in a parametric model. We will be using it to determine the
# coefficients of a line that best fits a given set of points.
# Since we are using a parametric model we make assumptions about the functional form of the
# relation ship between the independent and the dependent variables and in this case we
# assume there is a linear relationship (linear regression does not have to model linear
# relationships) and try to estimate the intersect and slope.
# We use a singel feature for X

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

rng = np.random
#Training Data
X = np.asarray([3,4,5,6.1,6.3,2.88,8.89,5.62,6.9,1.97,8.22,9.81,4.83,7.27,5.14,3.08])
Y = np.asarray([0.9,1.6,1.9,2.9,1.54,1.43,3.06,2.36,2.3,1.11,2.57,3.15,1.5,2.64,2.20,1.21])

# Coefficients
# Slope = m
# Intercept = c
# Randomly initialize m and c as shared variables in Theano. Shared mean that c and m can be
# used by multiple functions (they have global scope?). Shared variables are the fastes type
# in theano because the GPU can use different optimizations and since c and m are what will
# be updated alot it is a smart idea to have them as shared.

# In Theano thear are symbolic variables and expressions.
m_value = rng.rand()
c_value = rng.rand()

m = theano.shared(m_value, name = 'm')
c = theano.shared(c_value, name = 'c')

x = T.vector('X')
y = T.vector('y')

num_samples = X.shape[0]

# Defining the cost function
# Every machine learning model has a cost parameter which the ml algorithm tries to minimize.
# The cost parameter is essentially the cumulative error predicted by the model for the
# current set of coefficients.

# TODO also use RSS is there a difference?

# Use the randomly assigned values of c and m to predict y
prediction = T.dot(x,m) + c # our line function
# How big is the error in all the samples in x?
cost = T.sum(T.pow(prediction-y,2))/(2*num_samples) # MSE cost

# Optimization
# We use gradient decent
gradm = T.grad(cost,m)
gradc = T.grad(cost,c)

learning_rate = 0.01
training_steps = 10

train = theano.function([x,y],cost,updates = [(m,m-learning_rate*gradm), (c,c-learning_rate*gradc)])
test = theano.function([x],prediction)

estimates = np.zeros(training_steps)
x_vals = np.zeros(training_steps)
for i in range(training_steps):
    estimates[i] = train(X,Y)
    x_vals[i] = i


print("Slope :")
print(m.get_value()) # since m and c are shared we have to use get_value
print("Intercept :")
print(c.get_value())

axes = plt.gca()
axes.set_ylim([np.amin(estimates), np.amax(estimates)])
plt.plot(x_vals, estimates)
plt.show()

# Note: X and Y are the input data vecors while x and y are the Theano variables.
# All the expressions and functions are written in terms of the symbolc variables x and y while
# X and Y is used when calling the functions.
# /Users/henke/anaconda2/envs/theanohacking/bin/frameworkpython -m IPython
