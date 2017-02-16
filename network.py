"""
Modified from:
network3.py

by Michael Nielsen
neuralnetworksanddeeplearning.com/chap6.html
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv3d2d import conv3d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
import pool_ext as pool
from nipy import load_image #, save_image
import cPickle



#### Constants
#These should be set in the .theanorc file
#GPU = False
#
#if GPU:
#    print "Trying to run under a GPU.  If this is not desired, then modify "+\
#        "network.py\nto set the GPU flag to False."
#    try: theano.config.device = 'gpu'
#    except: pass # it's already set
#    theano.config.floatX = 'float32'
#else:
#    print "Running with a CPU.  If this is not desired, then the modify "+\
#        "network.py to set\nthe GPU flag to True."
#

#theano.config.exception_verbosity='high'
#theano.config.traceback.limit = 32

#### Load the data

def load_data_shared(TRAIN_STACK, TRAIN_LABELS,
                     VALID_STACK, VALID_LABELS,
                     TEST_STACK, TEST_LABELS):

    def get_nii_data(nifti_file):
        image = load_image(nifti_file)
        data = image.get_data().transpose(3, 0, 1, 2)
        print "data shape:"
        print data.shape
        return data

    def get_text_data(text_file):
        vector = []
        with open(text_file) as f:
            for line in f:
                vector.append(line.rstrip())
        return np.asarray(vector)


    def shared(DATA, LABELS):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        Shuffles index first to hopefully improve mini batch learning

        """
        idx = np.arange(len(DATA))
        np.random.shuffle(idx)

        shared_x = theano.shared(
            np.asarray(DATA[idx], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(LABELS[idx], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return [shared(get_nii_data(TRAIN_STACK),get_text_data(TRAIN_LABELS)),
            shared(get_nii_data(VALID_STACK), get_text_data(VALID_LABELS)),
            shared(get_nii_data(TEST_STACK), get_text_data(TEST_LABELS))
            ]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size, params_file, logfile, restart=False):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params_file = params_file
        if restart:
            print("Loading existing parameters")
            self.load_params()
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.tensor4("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.logout = open(logfile, 'a')

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        self.logout.write('Epoch,Training_Error,Validation_Error,Test_Error,Mean_Epoch_Cost\n')

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        training_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    epoch_cost = np.mean(
                        [train_mb(j) for j in xrange(num_test_batches)])
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    training_accuracy = np.mean(
                                [training_mb_accuracy(j) for j in xrange(num_training_batches)])
                    print("\nEpoch {0}\n".format(epoch)+\
                          "training accuracy: {0:.2%}\n".format(training_accuracy)+\
                          "validation accuracy: {0:.2%}\n".format(validation_accuracy)+\
                          "Mean epoch cost: {0:.3}\n".format(epoch_cost))
                    test_accuracy = 'NA'
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        self.save_params()
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])

                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
                    self.logout.write('{0},{1},{2},{3},{4}\n'.format(
                                                                  epoch,
                                                                  training_accuracy,
                                                                  validation_accuracy,
                                                                  test_accuracy,
                                                                  epoch_cost,'\n'))
                    self.logout.flush()

        print("Finished training network.")
        self.logout.close()
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

    def classify(self, data):
        classify_x, classify_y = data
        i = T.lscalar()
        batches = 22/self.mini_batch_size
        #print("data size: {0}".format(data_size))

        self.predictions = theano.function(
           [i],
           self.layers[-1].y_out,
           givens={
               self.x:
               classify_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]})
#               classify_x[i:-1]})


#        predictions = [self.predictions(j) for j in xrange(batches)]
        predictions = [self.predictions(0)]

        print(predictions)
        print(batches)
        #out_printed = theano.printing.Print()(self.layers[-1].y_out)


    def save_params(self):
        params = [layer.__getstate__() for layer in self.layers]
        f = open(self.params_file, 'wb')
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load_params(self):
        f = open(self.params_file, 'rb')
        params = cPickle.load(f)
        [layer.__setstate__(state) for layer,state in zip(self.layers, params)]
        f.close()


#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2, 2),
                 activation_fn="Sigmoid"):
        """`filter_shape` is a tuple of length 4, whose entries are the number
    [MaI    of filters, the number of input feature maps, the filter height, and the
        filter width.

        NUMBER OF FILTERS: how many new feature maps
        INPUT FEATURE MAPS: self explanatory
        FILTER H,W,D: H,W,D of local receptive field

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps / dimensions, the image
        height, width, and depth.

        `poolsize` is a tuple of length 3, whose entries are the z, y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = get_activation(activation_fn)
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        n_in = np.prod(image_shape[1:])

        if activation_fn == "Sigmoid" or activation_fn == "Tanh":
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                    dtype=theano.config.floatX),
                name="conv_w",
                borrow=True)
            self.b = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                    dtype=theano.config.floatX),
                name="conv_b",
                borrow=True)
            self.params = [self.w, self.b]

        elif activation_fn == "ReLU" or activation_fn == "lReLU":
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=np.sqrt(2.0/n_in), size=filter_shape),
                    dtype=theano.config.floatX),
                name="conv_w",
                borrow=True)
            self.b = theano.shared(
                np.zeros((filter_shape[0],),
                    dtype=theano.config.floatX)+0.01,
                name="conv_b",
                borrow=True)
            self.params = [self.w, self.b]

        else:
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=0.01, size=filter_shape),
                    dtype=theano.config.floatX),
                name="conv_w",
                borrow=True)
            self.b = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
                name="conv_b",
            borrow=True)
            self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        ## conv3d takes as input (Batch, Z, n_feature maps, Y, X)
        ## we feed it (Batch, n_feature_maps, X, Y, Z) so we need to shuffle it
        #OUTPUTS: (N, Z- z_filter + 1, n_features, Y - y_filter + 1, X - x_filter + 1)
        conv_out = conv3d(
            signals = self.inpt.dimshuffle(0,4,1,3,2),
            filters = self.w.dimshuffle(0,4,1,3,2),
            filters_shape = [self.filter_shape[idx] for idx in [0,4,1,3,2]],
            signals_shape = [self.image_shape[idx] for idx in [0,4,1,3,2]])
        conv_out = conv_out.dimshuffle(0, 2, 4, 3, 1)
        pooled_out = pool.max_pool_3d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x', 'x')) ##dimshuffle broadcasts the bias vector
                                                              ## across the 3D tensor dimvs
        self.output_dropout = self.output # no dropout in the convolutional layers


    def __getstate__(self):
        return (self.w, self.b)

    def __setstate__(self,state):
        W,b = state
        self.w = W
        self.b = b
        self.params = [self.w, self.b]


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn="Sigmoid", p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = get_activation(activation_fn)
        self.p_dropout = p_dropout
        # Initialize weights and biases

        # RANDOMLY INITIATED, optimized for sigmoid
        if activation_fn == "Sigmoid" or activation_fn == "Tanh":
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(
                        loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                    dtype=theano.config.floatX),
                name='full_w', borrow=True)

            self.b = theano.shared(
                np.asarray(np.random.normal(
                         loc=0.0, scale=1.0, size=(n_out,)),
                           dtype=theano.config.floatX),
                name='full_b',borrow=True)

            self.params = [self.w, self.b]

        elif activation_fn == "ReLU" or activation_fn == "lReLU":
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=np.sqrt(2.0/n_in), size=(n_in,n_out)),
                    dtype=theano.config.floatX),
                name="conv_w",
                borrow=True)
            self.b = theano.shared(
                np.zeros((n_out,),
                    dtype=theano.config.floatX)+0.01,
                name="conv_b",
                borrow=True)
            self.params = [self.w, self.b]

        else:
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(
                        loc=0.0, scale=0.01, size=(n_in, n_out)),
                    dtype=theano.config.floatX),
                name='full_w', borrow=True)
            self.b = theano.shared(
                np.asarray(np.random.normal(
                    loc=0.0, scale=1.0, size=(n_out,)),
                    dtype=theano.config.floatX),
                name='full_b', borrow=True)
            self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b) #weigh activation by dropout rate
        self.y_out = T.argmax(self.output, axis=1)

        #We use inpt_droput during training, whether we want the option or not.
        #the function randomly removes neurons from the network (using masked arrays)

        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def __getstate__(self):
        return (self.w, self.b)

    def __setstate__(self,state):
        W,b = state
        self.w = W
        self.b = b
        self.params = [self.w, self.b]


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='soft_w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='soft_b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        """Return the log-likelihood cost.
		This function... is clever and simple
		The indexing [T.arange(net.y.shape[0]), net.y]
		returns only the softmax values of output_dropout that match
		the desired output (indexed in Y)
		The vector y will have shape[0] == number of training examples (N).
		So indexing output_dropout at [arange(N), y] returns a vector with
		each row of output_dropout, but only the column indexed by Y (i.e., the
		desired output)
		"""
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def __getstate__(self):
        return (self.w, self.b)

    def __setstate__(self,state):
        W,b = state
        self.w = W
        self.b = b
        self.params = [self.w, self.b]


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)




# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
def lReLU(z): return T.maxium(0.01*z, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

def get_activation(function):
    if function == "ReLU":
        return ReLU
    if function =="lReLU":
        return lReLU
    elif function == "Linear":
        return linear
    elif function == "Sigmoid":
        return sigmoid
    elif function == "Tanh":
        return tanh




