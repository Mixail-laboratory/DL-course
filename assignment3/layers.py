import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if predictions.ndim == 1:
        norm_pred = predictions - np.max(predictions)
        exp_sum = np.sum(np.exp(norm_pred))
    else:
        norm_pred = predictions - np.max(predictions, axis=1)[:, np.newaxis]
        exp_sum = np.sum(np.exp(norm_pred), axis=1)[:, np.newaxis]
    probs = np.exp(norm_pred) / exp_sum
    
    return probs


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * (W ** 2).sum()
    grad = 2 * reg_strength * W

    return loss, grad


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        target_index = target_index.flatten()
        str_index_arr = np.arange(target_index.shape[0])
        loss = -np.mean(np.log(probs[(str_index_arr, target_index)]))
    
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    if probs.ndim == 1:
        subtr = np.zeros_like(probs)
        subtr[target_index] = 1
        dprediction = probs - subtr
    else:
        batch_size = predictions.shape[0]
        str_index_arr = np.arange(target_index.shape[0])
        subtr = np.zeros_like(probs)
        subtr[(str_index_arr, target_index.flatten())] = 1
        dprediction = (probs - subtr) / batch_size
    
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        out = np.maximum(self.X, 0) 
        
        return out 
    
    def backward(self, d_out):
        copy_d_out = d_out.copy()
        copy_d_out[self.X <= 0] = 0
        d_result = copy_d_out
        
        return d_result
        
    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        out = self.X.dot(self.W.value) + self.B.value
        
        return out
    
    def backward(self, d_out):
        self.B.grad += d_out.sum(axis=0, keepdims=True)
        self.W.grad += self.X.T.dot(d_out)
        d_input = d_out.dot(self.W.value.T)      
        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = Param(X.copy())

        out_height = height + 1 - self.filter_size + 2 * self.padding
        out_width  = width  + 1 - self.filter_size + 2 * self.padding
        
        out = np.zeros([batch_size, out_height, out_width, self.out_channels])

        self.X.value = np.pad(self.X.value, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                slice_X_flat = self.X.value[:, y:y + self.filter_size, x:x + self.filter_size, :]
                slice_X_flat = slice_X_flat.reshape(batch_size, -1)
                W_flat = self.W.value.reshape(-1, self.out_channels)

                out[:, y, x, :] = slice_X_flat.dot(W_flat) + self.B.value

        return out
    

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_input = np.zeros_like(self.X.value)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        W_flat = self.W.value.reshape(-1, self.out_channels)
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                slice_X_flat = self.X.value[:, y:y + self.filter_size, x:x + self.filter_size, :]
                slice_X_flat = slice_X_flat.reshape(batch_size, -1)

                d_input[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.dot(d_out[:, y, x, :], W_flat.T) \
                    .reshape(batch_size, self.filter_size, self.filter_size, self.in_channels)

                self.W.grad += np.dot(slice_X_flat.T, d_out[:, y, x, :]) \
                    .reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0)
        
        return d_input[:, self.padding:height - self.padding, self.padding:width - self.padding, :]
    
        
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    
    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width  = (width  - self.pool_size) // self.stride + 1
        
        out = np.zeros([batch_size, out_height, out_width, channels])
        
        for y in range(out_height):
            for x in range(out_width):
                out[:, y, x, :] += np.amax(X[:, 
                                              y * self.stride:y * self.stride + self.pool_size,
                                              x * self.stride:x * self.stride + self.pool_size, 
                                              :], axis=(1, 2))
        
        return out

    
    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        
        d_input = np.zeros_like(self.X)
        
        batch_idxs = np.repeat(np.arange(batch_size), channels)
        channel_idxs = np.tile(np.arange(channels), batch_size)
        
        for y in range(out_height):
            for x in range(out_width):
                slice_X = self.X[:,
                                y * self.stride:y * self.stride + self.pool_size,
                                x * self.stride:x * self.stride + self.pool_size,
                                :].reshape(batch_size, -1, channels)
               
                max_idxs = np.argmax(slice_X, axis=1)

                slice_d_input = d_input[:,
                                        y * self.stride:y * self.stride + self.pool_size,
                                        x * self.stride:x * self.stride + self.pool_size,
                                        :].reshape(batch_size, -1, channels)
                
                slice_d_input[batch_idxs, max_idxs.flatten(), channel_idxs] = d_out[batch_idxs, y, x, channel_idxs]

                d_input[:,
                        y * self.stride:y * self.stride + self.pool_size,
                        x * self.stride:x * self.stride + self.pool_size,
                        :] =\
                slice_d_input.reshape(batch_size, self.pool_size, self.pool_size, channels)
        
        return d_input
    
        
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = batch_size, height, width, channels
        return X.reshape(batch_size, -1)
        
    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
