# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:55:28 2020

@author: EduardoFelipe
"""

# Import libraries
import numpy as np
from p import RBPerceptron
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

zeros = np.zeros(10)
ones = zeros + 1
targets = np.concatenate((zeros,ones))

# preparação dos dados
w1 = np.array([[0.1, 1.1],[6.8, 7.1],[-3.5,-4.1],[2.0, 2.7],[4.1, 2.8],[3.1, 5.0],[-0.8, -1.3],[0.9, 1.2],[5.0, 6.4],[3.9, 4.0]])
w2 = np.array([[7.0, 4.2],[-1.4, -4.3],[4.5, 0.0],[6.3, 1.6],[4.2, 1.9],[1.4, -3.2],[2.4, -4.0],[2.5, -6.1],[8.4, 3.7],[4.1, -2.2]])
w3 = np.array([[-3.0, -2.9],[0.5, 8.7],[2.9, 2.1],[-0.1, 5.2],[-4.0, 2.2],[-1.3, 3.7],[-3.4, 6.2],[-4.1, 3.4],[-5.1, 1.6],[1.9, 5.1]])
X1 = np.concatenate((w1,w2))
X2 = np.concatenate((w2,w3))
D = targets
# inicia Perceptron
# 20 epocas, 0.01 taxa de aprendizado
rbp = RBPerceptron(20, 0.01)

# Treina Perceptron para X1
#trained_model = rbp.train(X1, D)

# Treina Perceptron para X2
trained_model = rbp.train(X2, D)


graph = plot_decision_regions(X2, D.astype(np.integer), clf=trained_model)
plt.title('Perceptron de Rosenblatt')
plt.xlabel('X1')
plt.ylabel('X2')

handles, labels = graph.get_legend_handles_labels()
graph.legend(handles, 
          ['w2', 'w3'], 
           framealpha=0.3, scatterpoints=1)


plt.show()
