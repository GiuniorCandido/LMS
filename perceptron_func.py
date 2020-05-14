# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:30:03 2020

@author: EduardoFelipe
"""

import numpy as np
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

# Implementação do Percepton de Rosenblatt
class RBPerceptron:

  # Constructor
  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate

  # Gera estimativa
  def predict(self, sample):
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    return np.where(outcome > 0, 1, 0)

  # treinamento
  def train(self, X, D):
    print("i [b        w1     w2]")
    # Inicia vetor de pesos nulos
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    print(0,self.w)
    # for numero de epocas
    for i in range(self.number_of_epochs):
      # For cada combinacao de x e target
      for sample, desired_outcome in zip(X, D):
        # gera estimatica e compara com target
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
        # Novo peso com a regra de aprendizagem do percepton
        weight_update = self.learning_rate * difference
        self.w[1:]    += weight_update * sample
        self.w[0]     += weight_update
      print(i+1,self.w) 
    return self

# inicia Perceptron
# 20 epocas, 0.01 taxa de aprendizado
rbp = RBPerceptron(20, 0.01)

# Treina Perceptron para X1
#trained_model = rbp.train(X1, D)

# Treina Perceptron para X2
trained_model = rbp.train(X2, D)


#plotagem dos gráficos
graph = plot_decision_regions(X2, D.astype(np.integer), clf=trained_model)
plt.title('Perceptron de Rosenblatt')
plt.xlabel('X1')
plt.ylabel('X2')

handles, labels = graph.get_legend_handles_labels()
graph.legend(handles, 
          ['w2', 'w3'], 
           framealpha=0.3, scatterpoints=1)


plt.show()
