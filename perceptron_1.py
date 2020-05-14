# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:30:03 2020

@author: EduardoFelipe
"""

import numpy as np

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

