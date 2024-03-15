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
    return 2 * input_array[:, 0:1] # x[:,0:1] will return all row and first column only -> dimension will be (m, 1). DON'T USE x[:,0] TO AVOID GETTING THE FINAL DIM -> (m,)

bc = dde.icbc.DirichletBC(geomtime, lambda input_array: double_first_column(input_array), lambda _, on_boundary: on_boundary)


