import numpy as np


class Perceptron:
  def __init__(self,eta,epochs):
    self.weights = np.random.randn(3)*1e-4
    print(f"initial weights before training:{self.weights}")
    self.eta = eta
    self.epochs = epochs

  def activationFunction(self,inputs,weights):
    z = np.dot(inputs,weights)
    return np.where(z>0,1,0)

  def fit(self, x,y):
    self.x = x
    self.y = y

    x_with_bias = np.c_[x,-np.ones((len(x),1))]
    print(f"x with bias : {x_with_bias}")

    for epoch in range(self.epochs):
      print("__"*10)
      print(f"for epoch :{epoch}")
      print("__"*10)

      y_hat = self.activationFunction(x_with_bias,self.weights)
      print(f"predicted value after forward pass: \n{y_hat}")

      self.error = self.y - y_hat
      print(f"error : \n{self.error}")

      self.weights = self.weights + self.eta * np.dot(x_with_bias.T,self.error)
      print(f"updated weights after epoch :\n{epoch}/{self.epochs} :\n{self.weights}")
      print("########"*10)
      
  def predict(self,x):
    x_with_bias = np.c_[x,-np.ones((len(x),1))]
    return self.activationFunction(x_with_bias,self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss: {total_loss}")