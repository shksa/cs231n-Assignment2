import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    # Conv layer
    # The parameters of the conv is of size (F, C, HH, WW) with
    # F give the no of filters, C,HH,WW characterize the size of
    # each filter
    # Input size : (N, C, H, W)
    # Output size : (N, F, H_conv, W_conv)
    C, H, W = input_dim
    F = num_filters

    filter_height = filter_size
    filter_width = filter_size

    stride_conv = 1  # stride
    P = (filter_size - 1) // 2  # pad

    # Height and width of the output volume of the conv layer
    H_conv = 1 + (H + 2 * P - filter_height) // stride_conv 
    W_conv = 1 + (W + 2 * P - filter_width) // stride_conv

    W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
    b1 = np.zeros(F)
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    

    # Pool layer : 2*2
    # The pool layer has no parameters but is important in the
    # count of dimension.
    # Input : (N, F, H_conv, W_conv)
    # Ouput : (N, F, H_pool, W_pool)

    width_pool = 2
    height_pool = 2
    stride_pool = 2
    H_pool = (H_conv - height_pool) // stride_pool + 1
    W_pool = (W_conv - width_pool) // stride_pool + 1

    # Hidden Affine layer
    # Size of the parameter (F*H_pool*W_pool, hidden_dim)
    # Input: (N, F*H_pool*W_pool)
    # Output: (N, hidden_dim)

    W2 = weight_scale * np.random.randn(F * H_pool * W_pool, hidden_dim)
    b2 = np.zeros(hidden_dim)

    # Output affine layer
    # Size of the parameter (hidden_dim, num_classes)
    # Input: (N, hidden_dim)
    # Output: (N, num_classes)

    W3 = weight_scale * np.random.randn(hidden_dim, num_classes)
    b3 = np.zeros(num_classes)

    self.params.update({'W1': W1,
                        'W2': W2,
                        'W3': W3,
                        'b1': b1,
                        'b2': b2,
                        'b3': b3})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    # Forward into the "conv - relu - pool" layer
    x = X
    w = W1
    b = b1
    crp_layer_output, cache_crp_layer = conv_relu_pool_forward(
        x, w, b, conv_param, pool_param)
    N, F, H_pool, W_pool = crp_layer_output.shape  # output shape

    # Forward into the "affine - relu" hidden layer
    x = crp_layer_output.reshape((N, F * H_pool * W_pool))
    w = W2
    b = b2
    hidden_layer_output, cache_hidden_layer = affine_relu_forward(x, w, b)
    N, Hh = hidden_layer_output.shape

    # Forward into the "affine" linear output layer
    x = hidden_layer_output
    w = W3
    b = b3
    scores, cache_output_layer = affine_forward(x, w, b)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W3**2)
    loss = data_loss + reg_loss

    # Backpropagation
    
    # Backprop into output affine layer
    dx3, dW3, db3 = affine_backward(dscores, cache_output_layer)
    dW3 += self.reg * W3

    # Backprop into affine - relu layer
    dx2, dW2, db2 = affine_relu_backward(dx3, cache_hidden_layer)
    dW2 += self.reg * W2

    # Backprop into the conv - relu - pool layer
    dx2 = dx2.reshape(N, F, H_pool, W_pool)
    dx1, dW1, db1 = conv_relu_pool_backward(dx2, cache_crp_layer)
    dW1 += self.reg * W1

    grads.update({'W1': dW1,
                  'b1': db1,
                  'W2': dW2,
                  'b2': db2,
                  'W3': dW3,
                  'b3': db3})
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

