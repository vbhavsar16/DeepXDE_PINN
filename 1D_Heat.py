import deepxde as dde
from deepxde.backend import tf
import numpy as np
from matplotlib import pyplot as plt

k = 0.4                                                                     # constant
L = 1
n = 1

geom = dde.geometry.Interval(0,L)
timedomain = dde.geometry.TimeDomain(0,n)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)                     # Making a domain of space and time

ic = dde.icbc.IC(geomtime, lambda x: np.sin(n* np.pi * x[:, 0:1]/L), lambda _, on_initial: on_initial) # x[:,0:1] will return all row and first column only -> dimension will be (m, 1). DON'T USE x[:,0] TO AVOID GETTING THE FINAL DIM -> (m,)

def double_first_column(input_array):
    return 2 * input_array[:, 0:1]                                          # x[:,0:1] will return all row and first column only -> dimension will be (m, 1). DON'T USE x[:,0] TO AVOID GETTING THE FINAL DIM -> (m,)

bc = dde.icbc.DirichletBC(geomtime, lambda input_array: double_first_column(input_array), lambda _, on_boundary: on_boundary)

def pde(comp,u):
    du_t = dde.grad.jacobian(u, comp, i=0, j=1)                             # Taking a jacobian of u (temperature/ x-directional velocity = scalar) over computational space and time. i,j are the value you choose to get the right value from the jacobian matrix.
    du_xx = dde.grad.hessian(u, comp, i=0, j=0)                             # Same way hessian of u and i, j value to choose the right value from the matrix. (i = row, j = column)
    return du_t - k * du_xx


data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc,ic],
    num_domain= 2540,
    num_boundary= 80,
    num_initial= 160,
    num_test= 2540
)

net = dde.nn.FNN([2] + [20]*3 + [1], "tanh", "Glorot normal")              # Syntex to define the net, 2 neuron + 3 hidden layers of 20 neurons + last layer with 1 neuron

#--------- Visulize the generated domain points ------------- #

plt.figure(figsize=(10,8))
plt.scatter(data.train_x_all[:,0], data.train_x_all[:,1])                  # data.train_x_all gives all the points of size (n,2), columns are x and t
plt.xlabel('x')
plt.ylabel('t')
plt.title('Domain points generated')
plt.show()

