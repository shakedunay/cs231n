import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  x_rsp = x.reshape(N, -1)
  out = x_rsp.dot(w) + b
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  N = x.shape[0]  
  x_rsp = x.reshape(N , -1)  
  dx = dout.dot(w.T)
  dx = dx.reshape(*x.shape)
  dw = x_rsp.T.dot(dout)
  db = np.sum(dout, axis = 0)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = x * (x >= 0)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = (x >= 0) * dout
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_mean = np.mean(x, axis = 0)
    sample_var = np.var(x , axis = 0)
    x_hat = (x - sample_mean) / (np.sqrt(sample_var  + eps))
    out = gamma * x_hat + beta
    cache = (gamma, x, sample_mean, sample_var, eps, x_hat)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    #pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    scale = gamma / (np.sqrt(running_var  + eps))
    out = x * scale + (beta - running_mean * scale)
    #pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  gamma, x, u_b, sigma_squared_b, eps, x_hat = cache
  N = x.shape[0]

  dx_1 = gamma * dout
  dx_2_b = np.sum((x - u_b) * dx_1, axis=0)
  dx_2_a = ((sigma_squared_b + eps) ** -0.5) * dx_1
  dx_3_b = (-0.5) * ((sigma_squared_b + eps) ** -1.5) * dx_2_b
  dx_4_b = dx_3_b * 1
  dx_5_b = np.ones_like(x) / N * dx_4_b
  dx_6_b = 2 * (x - u_b) * dx_5_b
  dx_7_a = dx_6_b * 1 + dx_2_a * 1
  dx_7_b = dx_6_b * 1 + dx_2_a * 1
  dx_8_b = -1 * np.sum(dx_7_b, axis=0)
  dx_9_b = np.ones_like(x) / N * dx_8_b
  dx_10 = dx_9_b + dx_7_a

  dgamma = np.sum(x_hat * dout, axis=0)
  dbeta = np.sum(dout, axis=0)
  dx = dx_10
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  gamma, x, sample_mean, sample_var, eps, x_hat = cache
  N = x.shape[0]
  dx_hat = dout * gamma
  dvar = np.sum(dx_hat* (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis = 0)
  dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var +eps), axis = 0) + dvar * np.mean(-2 * (x - sample_mean), axis =0)
  dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x-sample_mean) + 1.0 / N * dmean
  dgamma = np.sum(x_hat * dout, axis = 0)
  dbeta = np.sum(dout , axis = 0)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) >= p) / (1 - p)
    #mask = (np.random.rand(x.shape[1]) >= p) / (1 - p)
    out = x * mask
    #pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    #pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    #pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']

  x_pad = np.pad(
    array=x,
    pad_width=[
      (0,), # no pad for N 
      (0,), # no pad for C
      (pad,),
      (pad,),
    ],
    mode='constant',
    constant_values=0,
  )
  H_tag = 1 + (H + 2 * pad - HH) / stride
  W_tag = 1 + (W + 2 * pad - WW) / stride
  H_tag = int(H_tag)
  W_tag = int(H_tag)
  out = np.zeros((N, F, H_tag, W_tag))
  
  for i in range(H_tag):
    for j in range(W_tag):
        i_start = i * stride
        i_end = i_start + HH
        j_start = j * stride
        j_end = j_start + WW

        # mask for all samples and all colors
        x_pad_mask = x_pad[
          :, # all samples
          :, # all channels
          i_start:i_end,# HH
          j_start:j_end # WW
        ]
        assert x_pad_mask.shape == (N, C, HH, WW)
        for k in range(F):
          b_k = b[k]
          w_k = w[k, :, :, :]

          assert w_k.shape == (C, HH, WW)
          # each kernel has width height and channel(channel set by input x)
          w_k = w_k.reshape(1, C, HH, WW)
          # broadcast scalar mul to all samples
          conv_scalar_mul = x_pad_mask * w_k

          # sum color, width height for all samples
          conv = np.sum(
            conv_scalar_mul,
            axis=(
              1, # color
              2, # width
              3, # height
            ),
          )
          out[:, k, i, j] = conv + b_k
    # print(conv.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)

  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.
  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  (x, w, b, conv_param) = cache
  (N, C, H, W)   = x.shape
  (F, C, HH, WW) = w.shape
  stride = conv_param['stride']
  pad    = conv_param['pad']
  H_dash = 1 + (H + 2 * pad - HH) / stride
  H_dash = int(H_dash)
  W_dash = 1 + (W + 2 * pad - WW) / stride
  W_dash = int(W_dash)
  # out = np.zeros((N, F, H_dash, W_dash))
  dw = np.zeros(w.shape)
  x_new = np.lib.pad(x, ((1,1)), 'constant', constant_values=(0))
  x_new = x_new[1:-1,1:-1,:,:] #after zero padding
  dx = np.zeros(x_new.shape)

  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  for f in range(0,F): # on the filter dimension
        for c in range(0,C): # on the number of channels
            for k in range(0,W_dash): #along width of out
                for l in range(0,H_dash):  #along height of out
                    for n in range(0,N):
                        # out[n,f,k,l] += np.sum(x_new[n,c,k*stride:k*stride+WW,l*stride:l*stride+HH]*w[f,c,:,:])
                        dw[f,c,:,:] += dout[n,f,k,l]*x_new[n,c,k*stride:k*stride+WW,l*stride:l*stride+HH]
  for n in range(0,N):
      for f in range(0,F): # on the filter dimension
          for k in range(0,W_dash): #along width of out
            for l in range(0,H_dash):  #along height of out
                for c in range(0,C): # on the number of channels
                    dx[n, c, k*stride:k*stride+WW, l*stride:l*stride+HH] += dout[n,f,k,l]*w[f,c,:,:]
  dx = dx[:,:,1:-1,1:-1]
  db = np.sum(np.transpose(dout, [1,0,2,3]), axis = (1,2,3))

  return dx, dw, db

# from .im2col import im2col_indices, col2im_indices
# def conv_forward_naive_shaked(x, w, b, conv_param):
#   N, C, H, W   = x.shape
#   F, _, HH, WW = w.shape
#   stride, pad  = conv_param['stride'], conv_param['pad']

#   # Dimensionality check
#   assert ( H + 2 * pad - HH) % stride == 0, 'width doesn\'t work with current paramter setting'
#   assert ( W + 2 * pad - WW) % stride == 0, 'height doesn\'t work with current paramter setting'

#   # Initialize output
#   x_pad = np.pad(
#   array=x,
#   pad_width=[
#       (0,), # no pad for N 
#       (0,), # no pad for C
#       (pad,),
#       (pad,),
#   ],
#   mode='constant',
#   constant_values=0,
#   )
#   H_tag = 1 + (H + 2 * pad - HH) / stride
#   W_tag = 1 + (W + 2 * pad - WW) / stride
#   H_tag = int(H_tag)
#   W_tag = int(H_tag)

#   out = np.zeros((N, F, H_tag, W_tag))

#   for n_sample in range(N):
#     sample = x_pad[n_sample]
#     for k in range(F):
#       kernel = w[k]
#       b_k = b[k]
#       for i in range(H_tag):
#         for j in range(W_tag):
#           current_sum = 0
#           for hh in range(HH):
#             for ww in range(WW):
#               for c in range(C):
#                 one = kernel[c,hh,ww]
#                 two = sample[c, i-hh,j-ww]
#                 current_sum += one*two
#           out[n_sample, k, i, j] = current_sum + b_k

#   cache = (x, w, b, conv_param)
#   return out, cache

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.
  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.
  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.
  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  (N, C, H, W)   = x.shape
  (F, C, HH, WW) = w.shape
  stride = conv_param['stride']
  pad    = conv_param['pad']
  H_dash = 1 + (H + 2 * pad - HH) / stride
  H_dash = int(H_dash)
  W_dash = 1 + (W + 2 * pad - WW) / stride
  W_dash = int(W_dash)
  out = np.zeros((N, F, H_dash, W_dash))
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  x_new = np.pad(
    array=x,
    pad_width=[
        (0,), # no pad for N 
        (0,), # no pad for C
        (pad,),
        (pad,),
    ],
    mode='constant',
    constant_values=0,
  )
  for n in range(N):
    for k in range(F):
      for i in range(H_dash):
        for j in range(W_dash):
          current_sum = 0
          for c in range(C):
            j_start = j * stride
            j_end = j_start + WW
            i_start = i * stride
            i_end = i_start + HH

            x_new_window = x_new[
              n,
              c,  
              i_start:i_end,
              j_start:j_end,
            ]
            kernel = w[k,c,:,:]
            current_sum += np.sum(
              x_new_window * kernel,
            )

          out[n,k,i,j] = current_sum + b[k]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache



def conv_backward_naive_(dout, cache):
    X, W, b, conv_param, X_col = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter,
                        padding=pad, stride=stride)

    return dX, dW, db

def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  H_out = (H-HH)/stride+1
  W_out = (W-WW)/stride+1
  out = np.zeros((N,C,H_out,W_out))
  for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
            out[:,:,i,j] = np.max(x_masked, axis=(2,3)) 
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape
  HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  H_out = (H-HH)/stride+1
  W_out = (W-WW)/stride+1
  dx = np.zeros_like(x)
  
  for i in xrange(H_out):
     for j in xrange(W_out):
        x_masked = x[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW]
        max_x_masked = np.max(x_masked,axis=(2,3))
        temp_binary_mask = (x_masked == (max_x_masked)[:,:,None,None])
        dx[:,:,i*stride : i*stride+HH, j*stride : j*stride+WW] += temp_binary_mask * (dout[:,:,i,j])[:,:,None,None]
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  temp_output, cache = batchnorm_forward(x.transpose(0,3,2,1).reshape((N*H*W,C)), gamma, beta, bn_param)
  out = temp_output.reshape(N,W,H,C).transpose(0,3,2,1)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W = dout.shape
  dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape((N*H*W,C)),cache)
  dx = dx_temp.reshape(N,W,H,C).transpose(0,3,2,1)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
