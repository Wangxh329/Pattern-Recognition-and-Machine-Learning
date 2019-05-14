import tensorflow as tf
import numpy as np
from time import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
from PIL import Image

USE_GPU = True

###################################################################################################
#                                            PREAMBLE                                             #
###################################################################################################

log = lambda *args: print(datetime.now().strftime('%H:%M:%S'), ':', *args)

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test, mean_pixel, std_pixel


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0]
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))


X_train, y_train, X_val, y_val, X_test, y_test, mean_pixel, std_pixel = load_cifar10()
train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)

get_X_train_sample = lambda: next(iter(train_dset))[0][7] * std_pixel + mean_pixel

def select_device(use_gpu=True):
    from tensorflow.python.client import device_lib
    log(device_lib.list_local_devices())
    device = '/device:GPU:0' if use_gpu else '/CPU:0'
    log('Using device: ', device)
    return device

device = select_device(use_gpu=USE_GPU)

###################################################################################################
#                                              PART 1                                             #
###################################################################################################

def flatten(x):
    """
    Input:
    - TensorFlow Tensor of shape (N, D1, ..., DM)

    Output:
    - TensorFlow Tensor of shape (N, D1 * ... * DM)
    """
    x_flat = None
    ############################################################################
    # TODO: (1.a) Reshape tensor x into shape (N, D1 * ... * DM)               #
    ############################################################################

    x_flat = tf.reshape(x, (tf.shape(x)[0], -1))

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    return x_flat


def kaiming_normal(shape):
    """
    He et al, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification, ICCV 2015, https://arxiv.org/abs/1502.01852
    """
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)


def convnet_init():
    """
    Initialize the weights of a Three-Layer ConvNet, for use with the
    three_layer_convnet function defined above.
    """

    conv_w1 = tf.Variable(kaiming_normal([5, 5, 3, 32]))
    conv_b1 = tf.Variable(tf.zeros(32,))

    conv_w2 = tf.Variable(kaiming_normal([5, 5, 32, 32]))
    conv_b2 = tf.Variable(tf.zeros(32,))

    conv_w3 = tf.Variable(kaiming_normal([5, 5, 32, 64]))
    conv_b3 = tf.Variable(tf.zeros(64,))

    ############################################################################
    # TODO: (1.a), (2.a) Initialize the remaining parameters.                  #
    ############################################################################

#     conv_w4 = tf.Variable(kaiming_normal([4, 4, 64, 64]))
#     conv_b4 = tf.Variable(tf.zeros(64,))
    
#     # (1.a)
#     conv_w5 = tf.Variable(kaiming_normal([1, 1, 64, 10]))
#     conv_b5 = tf.Variable(tf.zeros(10,))
    
#     # (2.a.i)
#     conv_w5 = tf.Variable(kaiming_normal([16, 16, 32, 10]))
#     conv_b5 = tf.Variable(tf.zeros(10,))
    
#     # (2.a.ii)
#     conv_w5 = tf.Variable(kaiming_normal([8, 8, 32, 10]))
#     conv_b5 = tf.Variable(tf.zeros(10,))
    
    # (2.a.iii)
    conv_w5 = tf.Variable(kaiming_normal([4, 4, 64, 10]))
    conv_b5 = tf.Variable(tf.zeros(10,))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

#     params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w4, conv_b4, conv_w5, conv_b5]
#     params = [conv_w1, conv_b1, conv_w5, conv_b5]
#     params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w5, conv_b5]
    params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w5, conv_b5]

    return params


def convnet_forward(x, params):
    """
    A three-layer convolutional network.

    Args:
    - x: A TensorFlow Tensor of shape (N, H, W, 3) giving a minibatch of images
    - params: A list of TensorFlow Tensors giving the weights and biases for the network.
    """
#     [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w4, conv_b4, conv_w5, conv_b5] = params
#     [conv_w1, conv_b1, conv_w5, conv_b5] = params
#     [conv_w1, conv_b1, conv_w2, conv_b2, conv_w5, conv_b5] = params
    [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w5, conv_b5] = params

    # block 1 -> 16x16x32
    x1_1_pad = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0)
    x1_2_conv = tf.nn.conv2d(x1_1_pad, conv_w1, [1, 1, 1, 1], padding='VALID') + conv_b1
    x1_3_pad = tf.pad(x1_2_conv, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x1_4_pool = tf.nn.max_pool(x1_3_pad, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    x1_5_relu = tf.nn.relu(x1_4_pool)

    # block 2 -> 8x8x32
    x2_1_pad = tf.pad(x1_5_relu, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0)
    x2_2_conv = tf.nn.conv2d(x2_1_pad, conv_w2, [1, 1, 1, 1], padding='VALID') + conv_b2
    x2_3_relu = tf.nn.relu(x2_2_conv)
    x2_4_pad = tf.pad(x2_3_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x2_5_pool = tf.nn.avg_pool(x2_4_pad, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # block 3 -> 4x4x64
    x3_1_pad = tf.pad(x2_5_pool, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', constant_values=0)
    x3_2_conv = tf.nn.conv2d(x3_1_pad, conv_w3, [1, 1, 1, 1], padding='VALID') + conv_b3
    x3_3_relu = tf.nn.relu(x3_2_conv)
    x3_4_pad = tf.pad(x3_3_relu, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    x3_5_pool = tf.nn.avg_pool(x3_4_pad, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    ############################################################################
    # TODO: (1.a), (2.a) Implement the remaining forward pass.                 #
    ############################################################################

#     # block 4 -> 1x1x64
#     x4_1_conv = tf.nn.conv2d(x3_5_pool, conv_w4, [1, 1, 1, 1], padding='VALID') + conv_b4
#     x4_2_relu = tf.nn.relu(x4_1_conv)

#     # (1.a)
#     # block 5 -> 1x1x10
#     x5_1_conv = tf.nn.conv2d(x4_2_relu, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5
    
#     # (2.a.i)
#     # block 5 -> 1x1x10
#     x5_1_conv = tf.nn.conv2d(x1_5_relu, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5
    
#     # (2.a.ii)
#     # block 5 -> 1x1x10
#     x5_1_conv = tf.nn.conv2d(x2_5_pool, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5
    
    # (2.a.iii)
    # block 5 -> 1x1x10
    x5_1_conv = tf.nn.conv2d(x3_5_pool, conv_w5, [1, 1, 1, 1], padding='VALID') + conv_b5

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################

    logits = flatten(x5_1_conv)

    return logits


def three_layer_convnet_test():
    tf.reset_default_graph()

    with tf.device(device):
        x = tf.placeholder(tf.float32)

        # block 1
        conv_w1 = tf.zeros([5, 5, 3, 32])
        conv_b1 = tf.zeros(32)

        # block 2
        conv_w2 = tf.zeros([5, 5, 32, 32])
        conv_b2 = tf.zeros(32)

        # block 3
        conv_w3 = tf.zeros([5, 5, 32, 64])
        conv_b3 = tf.zeros(64)

        ############################################################################
        # TODO: (1.a), (2.a) Initialize the parameters.                            #
        ############################################################################

#         conv_w4 = tf.zeros([4, 4, 64, 64])
#         conv_b4 = tf.zeros(64)
        
#         # (1.a)
#         conv_w5 = tf.zeros([1, 1, 64, 10])
#         conv_b5 = tf.zeros(10)
        
#         # (2.a.i)
#         conv_w5 = tf.zeros([16, 16, 32, 10])
#         conv_b5 = tf.zeros(10)
        
#         # (2.a.ii)
#         conv_w5 = tf.zeros([8, 8, 32, 10])
#         conv_b5 = tf.zeros(10)

        # (2.a.iii)
        conv_w5 = tf.zeros([4, 4, 64, 10])
        conv_b5 = tf.zeros(10)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

#         params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w4, conv_b4, conv_w5, conv_b5]
#         params = [conv_w1, conv_b1, conv_w5, conv_b5]
#         params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w5, conv_b5]
        params = [conv_w1, conv_b1, conv_w2, conv_b2, conv_w3, conv_b3, conv_w5, conv_b5]
        logits = convnet_forward(x, params)

    # Inputs to convolutional layers are 4-dimensional arrays with shape [batch_size, height, width, channels]
    x_np = np.zeros((64, 32, 32, 3))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logits_np = sess.run(logits, feed_dict={x: x_np})
        log('logits_np has shape', format(logits_np.shape))


with tf.device('/cpu:0'):
    three_layer_convnet_test()


def training_step(logits, y, params, learning_rate):
    """
    Set up the part of the computational graph which makes a training step.

    Args:
    - logits: TensorFlow Tensor of shape (N, C) giving classification scores for
      the model.
    - y: TensorFlow Tensor of shape (N,) giving ground-truth labels for scores;
      y[i] == c means that c is the correct class for scores[i].
    - params: List of TensorFlow Tensors giving the weights of the model
    - learning_rate: Python scalar giving the learning rate to use for gradient
      descent step.

    Returns:
    - loss: A TensorFlow Tensor of shape () (scalar) giving the loss for this
      batch of data; evaluating the loss also performs a gradient descent step
      on params (see above).
    """
    # First compute the loss; the first line gives losses for each example in
    # the mini-batch, and the second averages the losses across the batch
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(losses)

    # Compute the gradient of the loss with respect to each parameter of the the
    # network. This is a very magical function call: TensorFlow internally
    # traverses the computational graph starting at loss backward to each element
    # of params, and uses back-propagation to figure out how to compute gradients;
    # it then adds new operations to the computational graph which compute the
    # requested gradients, and returns a list of TensorFlow Tensors that will
    # contain the requested gradients when evaluated.
    grad_params = tf.gradients(loss, params)

    # Make a gradient descent step on all of the model parameters.
    new_weights = []
    for w, grad_w in zip(params, grad_params):
        new_w = tf.assign_sub(w, learning_rate * grad_w)
        new_weights.append(new_w)

    # Insert a control dependency so that evaluating the loss causes a weight
    # update to happen.
    with tf.control_dependencies(new_weights):
        return tf.identity(loss)


def train(model_fn, init_fn, learning_rate, epochs, print_every=100):
    """
    Train a model on CIFAR-10.

    Args:
    - model_fn: A Python function that performs the forward pass of the model
      using TensorFlow; it should have the following signature:
      scores = model_fn(x, params) where x is a TensorFlow Tensor giving a
      minibatch of image data, params is a list of TensorFlow Tensors holding
      the model weights, and scores is a TensorFlow Tensor of shape (N, C)
      giving scores for all elements of x.
    - init_fn: A Python function that initializes the parameters of the model.
      It should have the signature params = init_fn() where params is a list
      of TensorFlow Tensors holding the (randomly initialized) weights of the
      model.
    - learning_rate: Python float giving the learning rate to use for SGD.
    """
    # First clear the default graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')
    # Set up the computational graph for performing forward and backward passes,
    # and weight updates.
    with tf.device(device):
        # Set up placeholders for the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])
        params = init_fn()  # Initialize the model parameters
        scores = model_fn(x, params)  # Forward pass of the model
        loss = training_step(scores, y, params, learning_rate)

    train_losses = []
    test_accuracies = []

    # Now we actually run the graph many times using the training data
    with tf.Session() as sess:
        # Initialize variables that will live in the graph
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            log('epoch {:>4d}/{:>4d}'.format(epoch, epochs))
            epoch_time = time()
            for t, (x_np, y_np) in enumerate(train_dset):
                # Run the graph on a batch of training data; recall that asking
                # TensorFlow to evaluate loss will cause an SGD step to happen.
                feed_dict = {x: x_np, y: y_np}
                loss_np = sess.run(loss, feed_dict=feed_dict)

                # Periodically print the loss and check accuracy on the val set
                if t % print_every == 0:
                    num_correct, num_samples, acc = get_accuracy(sess, val_dset, x, scores, is_training)
                    log('   iteration = {:>4d}, loss = {:>8.4f}, accuracy = {:>8.2f}%'.format(t, loss_np, acc))

            train_losses.append(loss_np)
            test_accuracies.append(acc)

            log('epoch {:>4d} took {:>.2f}s'.format(epoch, time()-epoch_time))

        return params, sess.run(params), train_losses, test_accuracies


def get_accuracy(sess, dset, x, logits, is_training=None):
    """
    Check accuracy on a classification model.

    Args:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.

    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(logits, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    return num_correct, num_samples, 100 * acc



############################################################################
# TODO: (1.b) Adjust learning-rate and number of epochs.                   #
############################################################################
learning_rate = 3e-3
epochs = 80
############################################################################
#                             END OF YOUR CODE                             #
############################################################################

params, params_val, train_losses, test_accuracies = train(convnet_forward, convnet_init, learning_rate, epochs=epochs)

############################################################################
# TODO: (1.c) Plot.                                                        #
############################################################################
# plot training loss vs epochs
plt.figure(figsize=(16,9))
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('Training Loss', fontsize = 18)
plt.title('Training Loss vs Epochs (learning rate = 3e-3)', fontsize = 23)
plt.plot(range(epochs), train_losses, 'o-', color='deeppink', lw=2)
plt.show()
plt.savefig('1_loss.png')

# plot accuracies vs epochs
plt.figure(figsize=(16,9))
plt.xlabel('Epochs', fontsize = 18)
plt.ylabel('Test Accuracy', fontsize = 18)
plt.title('Test Accuracy vs Epochs (learning rate = 3e-3)', fontsize = 23)
plt.plot(range(epochs), test_accuracies, 'o-', color='deeppink', lw=2)
plt.show()
plt.savefig('2_accuracy.png')


############################################################################
#                             END OF YOUR CODE                             #
############################################################################

