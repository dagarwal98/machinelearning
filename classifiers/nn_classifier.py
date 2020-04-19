import numpy as np
import matplotlib.pyplot as plt

class NeuralNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over 2 classes.
  We train the network with a sigmoid loss function and L2 regularization on the
  weight matrices. The network uses a sigmoid nonlinearity after the first fully
  connected layer.

    The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values.
    Weights are stored in the variable self.params, which is a dictionary 
    with the following keys:

    W1: First layer weights; has shape (D + 1, H)
    W2: Second layer weights; has shape (H + 1, 1)
    
    1 is added to the first dimension to account for the bias term.

    Inputs:
    - input_size: The dimension D of the input data (# of features).
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C. Here, it will be 1 for a binary class.
    """
    
    # set seed to get the same result for all runs
    np.random.seed(0)
    
    # initialize weights
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size + 1, hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size + 1, output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer value 0 or 1. This parameter is optional; if it is 
      not passed then we only return scores, and if it is passed then we instead 
      return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, 1) where scores[i] is
    the score for input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1 = self.params['W1']
    W2 = self.params['W2']

    # Compute the forward pass
    scores = None

    # Perform forward pass, 
    # Compute the class scores for the input. 
    # shape of a3 - (N, 1)
    a1, a2, a3 = self.feed_forward(X, W1, W2)
    scores = a3
    
    # If the targets are not given then return scores
    if y is None:
      return scores

    # Compute loss
    cost = None

    # Include data loss and L2 regularization for W1 and W2.
    # Use the Sigmoid classifier loss. 
    cost = self.loss_vector_reg(scores, y, W1, W2, reg)

    # Backward pass: compute gradients
    grads = {}

    # Compute backward pass, computing the derivatives of the weights. 
    D1, D2 = self.backprop(a1, W1, a2, W2, a3, y, reg)
    grads['W1'] = D1
    grads['W2'] = D2

    return cost, grads

  def add_bias(self, x):
    # add 1 for the bias unit
    return np.c_[np.ones(x.shape[0]), x] 

  def sigmoid(self, z):
    # sigmoid function
    return 1.0/(1.0 + np.exp(-z))

  def gprime(self, a):
    return a * (1 - a)

  def loss_vector_reg(self, a_out, y, w1, w2, regrate):
    # add epsilon to avoid division by zero error
    epsilon = 1e-8 
    m = y.size
    
    return -(1/m) * (np.sum(
                            np.dot(y.T, np.log(a_out + epsilon)) + 
                            np.dot((1-y).T, np.log(1-a_out + epsilon))
                            )
                     ) + \
        (regrate * np.sum(w1**2 + w2**2))

  def feed_forward(self, x, w1, w2):
    """
    x -- dimension: (N, 2)
    w1 -- dimension: (3, 2)
    w2 -- dimension: (3, 1)
    
    """
    # forward calculations
    
    # add bias unit to input matrix
    # a1: (N, 3)
    a1 = self.add_bias(x)

    # two units in layer 2
    # z2: (N, 3) . (3, 2) = (N, 2)
    z2 = np.dot(a1, w1)
    # apply sigmoid function
    # a2: (N, 2)
    a2 = self.sigmoid(z2)
   
    # one unit in layer 3
    # a2: (N, 3)
    a2 = self.add_bias(a2)
    # z3: (N, 3) . (3, 1) = (N, 1)
    z3 = np.dot(a2, w2)
    # apply sigmoid function
    # a3: (N, 1)
    a3 = self.sigmoid(z3)
    
    # a1: (N, 3)
    # a2: (N, 3)
    # a3: (N, 1)
    return a1, a2, a3

  def backprop(self, a1, w1, a2, w2, a3, y, regrate):
    """
    a1: (N, 3)
    w1: (3, 2)
    a2: (N, 3)
    w2: (3, 1)
    a3: (N, 1)
    y:  (N, 1)
    
    """
    m = y.shape[0]
    
    # set first row of weights to 0 (bias)
    w1[0,:] = 0
    w2[0,:] = 0 
    
    #output layer
    # d3: (N, 1)
    d3 = a3 - y
    # d2: (N, 1) . (1, 3) * (N, 3) = (N, 3)
    d2 = np.dot(d3, w2.T) * self.gprime(a2)
    
    # sinc a0 is bias term with no contribution to error from earlier part of the network. So
    # we don't need to backpropage it.
    # d2: (N, 2)
    d2 = d2[:, 1:]
    
    # Adding derivative of regularization term
    # D2: (3, N) . (N, 1) + (3, 1) = (3, 1)
    # D1: (3, N) . (N, 2) + (3, 2) = (3, 2)
    D2 = (np.dot(a2.T, d3) + (regrate/m) * w2)
    D1 = (np.dot(a1.T, d2) + (regrate/m) * w1)
    return D1, D2

  def train(self, X, y, X_val=None, y_val=None,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, 2) giving training data.
    - y: A numpy array f shape (N, 1) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < 1.
    - X_val: A numpy array of shape (N_val, 2) giving validation data.
    - y_val: A numpy array of shape (N_val, 1) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]

    # Use Gradient Descent to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):

      # Compute loss and gradients
      cost, grads = self.loss(X, y, reg=reg)
      loss_history.append(cost)

      # Use the gradients in the grads dictionary to update the
      # parameters of the network (stored in the dictionary self.params)
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, cost))

      # check train and val accuracy and decay learning rate.
      train_acc = (self.predict(X) == y).mean()
      val_acc = (self.predict(X_val) == y_val).mean()
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      # Decay learning rate
      learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history
    }

  def test(self, X, y, reg):
    W1 = self.params['W1']
    W2 = self.params['W2']
    _, _, a3_out = self.feed_forward(X, W1, W2)
    cost = self.loss_vector_reg(a3_out, y,  W1, W2, reg)
    accuracy = (self.predict(X) == y).mean()
    return {
     'cost': cost,
     'accuracy': accuracy,
     'estimated': a3_out > 0.5
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict score for the binary class.

    Inputs:
    - X: A numpy array of shape (N, 2) giving N 2-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N, 1) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where c is 1 if the score > 0.5, else it is 0.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    _, _, a3_est = self.feed_forward(X, self.params['W1'], self.params['W2'])
    y_pred = a3_est > 0.5
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


