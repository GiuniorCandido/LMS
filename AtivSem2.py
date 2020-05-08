# Andres E. M. Colognesi - 9838161

##############################
## Atividade 2 - Int. Comp. ##
##############################

## Importing libraries:

import numpy as np
import matplotlib.pyplot as plt


## Defining functions:

def classes(point):
    '''
    Receives a point and returns classification booleans. True means the point
    belongs to the corresponding class.

    Input: point [x1, x2]
    Output: booleans is_g1, is_g2, is_g3
    '''
    x1, x2 = point
    #Belongs to a class if the value is positive:
    is_g1 = (-x1 + x2) > 0
    is_g2 = (x1 + x2 - 5) > 0
    is_g3 = (-x2 + 1) > 0

    return is_g1, is_g2, is_g3

def sort(data):
    '''
    Receives a set of points [x1, x2] displated in lines and sorts them into the
    3 classes (g1, g2 and g3).

    Input: data (array of points [x1, x2])
    Output: sorted points numpy arrays g1, g2, g3, indet_mult and indet_none
    (indeterminate in multiple classes and indeterminate in no class)
    '''
    #Creating sorting arrays
    g1 = []
    g2 = []
    g3 = []
    indet_mult = []
    indet_none = []
    #Going through all lines of data:
    for i in range(data.shape[0]):
        point = data[i,:]
        is_g1, is_g2, is_g3 = classes(point) #evaluating classes for current point
        if sum([is_g1, is_g2, is_g3]) >= 2: #indeterminate multiple classes
            indet_mult.append(point)
        elif  sum([is_g1, is_g2, is_g3]) == 0: #indeterminate none
            indet_none.append(point)
        elif is_g1: #belongs only to class g1
            g1.append(point)
        elif is_g2: #belongs only to class g2
            g2.append(point)
        else: #belongs only to class g3
            g3.append(point)
    #Converting to numpy arrays and returning:
    return np.array(g1), np.array(g2), np.array(g3), np.array(indet_mult), np.array(indet_none)


## Main program:

r = 10 #limit range of interval
p = 100 #number of points in interval
data = []
#Creating square of data points inside legth x length:
for i in range(p):
    for j in range(p):
        data.append([r*i/p, r*j/p])

g1, g2, g3, indet_mult, indet_none = sort(np.array(data)) #sorting data into classes

#Plot results:

plt.scatter(g1[:,0], g1[:,1], c='green', label='g1')
plt.scatter(g2[:,0], g2[:,1], c='blue', label='g2')
plt.scatter(g3[:,0], g3[:,1], c='red', label='g3')
plt.scatter(indet_mult[:,0], indet_mult[:,1], c='yellow', label='Indeterminate (multiple classes)')
plt.scatter(indet_none[:,0], indet_none[:,1], c='black', label='Indeterminate (no class)')
plt.title("Classifications")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
