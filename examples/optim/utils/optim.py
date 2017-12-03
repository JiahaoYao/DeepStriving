import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum']*v-config['learning_rate']*dw
    next_w = w + v

    config['velocity'] = v

    return next_w, config



def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    config['cache'] = config['decay_rate']*config['cache']+(1-config['decay_rate'])*dx**2
    dx = dx/ np.sqrt(config['cache']+config['epsilon'])
    next_x = x - config['learning_rate']*dx

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    config['t'] +=1
    config['m'] = config['beta1']*config['m']+(1-config['beta1'])*dx
    config['v'] = config['beta2']*config['v']+(1-config['beta2'])*dx**2
    m_new = config['m']/(1-config['beta1']**config['t'])
    v_new = config['v']/(1-config['beta2']**config['t'])
    next_x = x - config['learning_rate']*m_new/(np.sqrt(v_new)+config['epsilon'])


    return next_x, config

# Reference: https://zhuanlan.zhihu.com/p/22252270
def adagrad(x, dx, config=None):
    """AdaGrad method, it regularizes the output."""

    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('eta', 0)
    next_x = None
    config['eta'] += dx**2
    next_x = x - config['learning_rate']*dx/ np.sqrt(config['eta'] +config['epsilon'])
    return next_x, config

def adadelta(x, dx, config=None):
    """adadelata method is also practical"""
    if config is None: config = {}
    config.setdefault('learning_rate', 1)
    config.setdefault('epsilon', 1e-6)
    config.setdefault('rho', 0.95)
    config.setdefault('eta', 0)
    config.setdefault('eta_dx', 0)

    next_x = None
    config['eta'] = config['rho']* config['eta'] + (1-config['rho'])*dx**2
    delta_x = - np.sqrt(config['eta_dx']+config['epsilon'])/np.sqrt(config['eta']+config['epsilon']) * dx
    config['eta_dx'] = config['eta_dx']*config['rho']-delta_x**2*(1-config['rho'])
    next_x = x + config['learning_rate']*delta_x
    return next_x, config

def adamax(x, dx, config=None):
    """adamax is the variants of the adam"""
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)
    
    next_x = None
    config['t'] +=1
    config['m'] = config['beta1']*config['m']+(1-config['beta1'])*dx
    config['v'] = np.maximum(config['beta2']*config['v'], abs(dx))
    m_new = config['m']/(1-config['beta1']**config['t'])

    next_x = x - config['learning_rate']*m_new/(np.sqrt(config['v']+config['epsilon']))
    
    
    return next_x, config

def nadam(x, dx, config=None):
    """ this is a adam with nesterov momentom"""
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.99)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)
    config.setdefault('mu', 1)
    
    next_x = None
    mu_t = (1 - .5 * .96 **(config['t']/250)) * config['beta1']
    config['mu'] *= mu_t
    config['t'] +=1
    mu_tt = (1 - .5 * .96 **(config['t']/250)) * config['beta1']
    dxx = dx/ (1- config['mu'])
    config['m'] = config['beta1']*config['m']+(1-config['beta1'])*dxx
    config['v'] = config['beta2']*config['v']+(1-config['beta2'])*dxx**2
    m_new = config['m']/(1- config['mu']*mu_tt)
    v_new = config['v']/(1-config['beta2']**config['t'])
    next_x = x - config['learning_rate']*m_new/(np.sqrt(v_new)+config['epsilon'])
    
    return next_x, config
                                                
                          





