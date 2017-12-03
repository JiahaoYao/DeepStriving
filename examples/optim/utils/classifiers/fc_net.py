from builtins import range
from builtins import object
import numpy as np

from utils.layers import *
from utils.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # Weights Initialization
        W1 = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        b2 = np.zeros(num_classes)

        self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X1, X1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        X2, X2_cache = relu_forward(X1)
        X3, X3_cache = affine_forward(X2, self.params['W2'], self.params['b2'])
        scores = X3

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, loss_gradient = softmax_loss(scores, y)
        reg_loss = self.reg*(np.sum(self.params['W1']**2)+np.sum(self.params['W2']**2))/2
        loss += reg_loss
        loss_gradient_3, dW2, db2 = affine_backward(loss_gradient, X3_cache)
        loss_gradient_2 = relu_backward(loss_gradient_3, X2_cache)
        loss_gradient_1, dW1, db1 = affine_backward(loss_gradient_2, X1_cache)
        dW2 += self.reg*self.params['W2']
        dW1 += self.reg*self.params['W1']

        grads = {'W2':dW2, 'b2':db2, 'W1':dW1, 'b1':db1}

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        self.params['W1'] = np.random.normal(0, weight_scale, size=(input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
        if use_batchnorm:
            self.params['gamma1'] = np.ones(hidden_dims[0])
            self.params['beta1'] = np.zeros(hidden_dims[0])
        for i in range(len(hidden_dims)-1):
            index = i+2
            self.params['W{}'.format(index)] = np.random.normal(0, weight_scale, size=(hidden_dims[i], hidden_dims[i+1]))
            self.params['b{}'.format(index)] = np.zeros(hidden_dims[i+1])
            if use_batchnorm:
                self.params['gamma{}'.format(index)] = np.ones(hidden_dims[i+1])
                self.params['beta{}'.format(index)] = np.zeros(hidden_dims[i+1])
        self.params['W{}'.format(self.num_layers)] = np.random.normal(0, weight_scale, size=(hidden_dims[-1], num_classes))
        self.params['b{}'.format(self.num_layers)] = np.zeros(num_classes)




        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        x = {}
        x_cache = {}
        # print(self.params)

        x['x1'], x_cache['x1'] = affine_forward(X, self.params['W1'], self.params['b1'])
        if self.use_batchnorm:
            x['x1_bn'], x_cache['x1_bn'] = batchnorm_forward(x['x1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])
        else:
            x['x1_bn'] = x['x1']
        x['x1_relu'], x_cache['x1_relu'] = relu_forward(x['x1_bn'])
        if self.use_dropout:
            x['x1_dropout'], x_cache['x1_dropout'] = dropout_forward(x['x1_relu'], self.dropout_param)
        else:
            x['x1_dropout'] = x['x1_relu']
        for i in range(self.num_layers-2):
            index = i+2
            x['x{}'.format(index)], x_cache['x{}'.format(index)] = affine_forward(x['x{}_dropout'.format(index-1)], self.params['W{}'.format(index)], self.params['b{}'.format(index)])
            if self.use_batchnorm:
                x['x{}_bn'.format(index)], x_cache['x{}_bn'.format(index)] = batchnorm_forward(x['x{}'.format(index)], self.params['gamma{}'.format(index)], self.params['beta{}'.format(index)],
                                                             self.bn_params[i+1])
            else:
                x['x{}_bn'.format(index)] = x['x{}'.format(index)]
            x['x{}_relu'.format(index)], x_cache['x{}_relu'.format(index)] = relu_forward(x['x{}_bn'.format(index)])
            if self.use_dropout:
                x['x{}_dropout'.format(index)], x_cache['x{}_dropout'.format(index)] = dropout_forward(x['x{}_relu'.format(index)], self.dropout_param)
            else:
                x['x{}_dropout'.format(index)] = x['x{}_relu'.format(index)]

        index = self.num_layers
        x['x{}'.format(index)], x_cache['x{}'.format(index)] = affine_forward(x['x{}_dropout'.format(index - 1)],
                                                                              self.params['W{}'.format(index)],
                                                                              self.params['b{}'.format(index)])
        scores = x['x{}'.format(index)]


        # If test mode return early
        if mode == 'test':
            return scores

        index = self.num_layers
        loss, grads = 0.0, {}
        loss, loss_gradient = softmax_loss(scores, y)
        for i in range(self.num_layers):
            j = i+1
            loss += self.reg*np.sum(self.params['W{}'.format(j)]**2)/2

        loss_gradient, grads['W{}'.format(index)], grads['b{}'.format(index)] = \
            affine_backward(loss_gradient, x_cache['x{}'.format(index)])
        grads['W{}'.format(index)] += self.reg* self.params['W{}'.format(index)]
        for i in range(self.num_layers-1,0,-1):
            index = i
            if self.use_dropout:
                loss_gradient = dropout_backward(loss_gradient, x_cache['x{}_dropout'.format(index)])
            loss_gradient = relu_backward(loss_gradient, x_cache['x{}_relu'.format(index)])
            if self.use_batchnorm:
                loss_gradient, grads['gamma{}'.format(index)], grads['beta{}'.format(index)] =\
                    batchnorm_backward(loss_gradient, x_cache['x{}_bn'.format(index)])
            loss_gradient, grads['W{}'.format(index)], grads['b{}'.format(index)] = \
                affine_backward(loss_gradient, x_cache['x{}'.format(index)])
            grads['W{}'.format(index)] += self.reg * self.params['W{}'.format(index)]


        return loss, grads
