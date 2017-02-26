import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
 
  # Using the formulas in http://cs231n.github.io/linear-classify/ and
  # slide 29 in lecture 3. 
  # scores = f(x,W) = XW
  # Probability p = exp(scores(k)) / sum(exp(scores(j)))
  # Loss(i) = log(y(i)*p(i)) = -log(p[y(i)])
  # or 
  # Loss(i) = -f(y(i)) + log(Sum(exp(f(i)))).
  #
  # To avoid numeric instability instead: 
  # first shift the values of f so that the highest number is 0:
  # f -= np.max(f) # f becomes [-666, -333, 0]
  # p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer

  for i in xrange(num_train):
      f = np.dot(X[i], W)
      f -= np.max(f) # ensuring numeric stability.  

      sumVal = 0
      for val in f:
          sumVal += np.exp(val)

      # At this point we can calculate the loss using two equations:
      # equation #1:
      # loss += -f[y[i]] + np.log(sumVal)
      # equation #2:
      p_y = np.exp(f[y[i]]) / sumVal
      loss += -np.log(p_y)

      # Using the formulas in http://cs231n.github.io/linear-classify/ we
      # calculate the drivative using http://cs231n.github.io/optimization-1/
      for j in xrange(num_classes):
          p = np.exp(f[j]) / sumVal
          dW[:,j] += (p - (j==y[i])) * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW   += 0.5 * 2 * reg * W   # derivative of 0.5*reg*np.sum(W^2) wrt w

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
 
  f = np.dot(X, W)
  f -= np.max(f)          # ensuring numeric stability.  
  f_exp = np.exp(f)
  p = f_exp/np.sum(f_exp, axis=1, keepdims=True)
  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  mask = np.zeros(p.shape)
  mask[np.arange(num_train), y] = 1
  dW = np.dot(X.T, (p-mask))     

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW   += 0.5 * 2 * reg * W   # derivative of 0.5*reg*np.sum(W^2) wrt w

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

