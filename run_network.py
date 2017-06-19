"""
Reads network parameters from json file.


## Layer syntax from code:
## image_shape = (mini batch size,
##                number of input dimensions/channels/feature maps from previous layer (or 1 for initial input),
##               Image X,Y,Z)

## Filter_shape= (number of filters i.e. how many new feature maps to create in this layer,
##                input feature maps: number of input dimensions/feature maps from previous layer (or initial input),
##                filter X,Y,Z: x,y,z of local receptive field)

## After a conv layer:
## Next layer input size will be:
## X - x_filter + 1, Y - y_filter + 1, Z - z_filter + 1
## Then divide by pool size
## For a fully connected layer, multiply these together with the number of feature maps in the previous layer
## (if following a ConvPool layer)
"""
import json
import network as nn
import time
import numpy as np
import sys,os
from nipy import load_image
import theano
import theano.tensor as T


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



class build_network(object):
    def __init__(self, network_file, classify=False):

        with open(network_file) as net_data:
            self.network = json.load(net_data)

        if classify:
            self.minibatch_size = 1
        else:
            self.minibatch_size = self.network['mini_batch_size']
        self.layers = [self.build_init(self.network['input_dims'])]
        self.layers_construct = []
        self.logfile =  os.path.dirname(network_file)+'/accuracy.csv'
        self.figfile=  os.path.dirname(network_file)+'/metrics.png'
        self.predictions = os.path.dirname(network_file)+'/predictions.csv'
        self.params_file = os.path.dirname(network_file)+'/params.save'
        self.restart = self.network['restart']

        for n_layer in xrange(len(self.network['Structure'])):
            lid = "layer_{0}".format(n_layer)
            layer =  self.network['Structure'][lid]

            if  layer['type'] == 'Conv':
                lout = self.build_conv(layer)

            elif layer['type'] == 'Full':
                lout = self.build_full(layer)

            elif layer['type'] == 'Softmax':
                lout = self.build_soft(layer)

            self.layers.append(lout)
            self.layers_construct.append(lout['construct'])





    def build_init(self, layer):
        out_shape = np.asarray(self.network['input_dims'])
        return {'construct' : "INIT",
                'out_dims' : out_shape,
                'params' : 0}



    def build_conv(self, layer):

        in_shape = self.layers[-1]['out_dims']
        feature_maps = np.asarray(layer['feature_maps'])
        image_shape=[[self.minibatch_size], in_shape.tolist()]
        filter_shape=([feature_maps.tolist()], [in_shape[0]], layer['lrf'])

        construct = nn.ConvPoolLayer(
            image_shape=tuple([param for sublist in image_shape for param in sublist]),
            filter_shape=tuple([param for sublist in filter_shape for param in sublist]),
            poolsize=tuple(layer['max_pool']),
            activation_fn=layer['activation_function'])

        out_shape = ((in_shape[1:] - np.asarray(layer['lrf'])) + 1) / np.asarray(layer['max_pool'])

        out_dims = np.append(feature_maps, out_shape)

        layer = {'construct':construct,
                 'out_dims' : out_dims,
                 'params' : tuple([param for sublist in filter_shape for param in sublist])
                 }
        return layer


    def build_full(self, layer):

        in_shape = np.product(self.layers[-1]['out_dims'])
        out_shape = np.asarray(layer['n_out'])

        construct = nn.FullyConnectedLayer(
            n_in= in_shape,
            n_out = layer['n_out'],
            activation_fn = layer['activation_function'],
            p_dropout = layer['p_dropout']
        )

        layer = {'construct' : construct,
                 'out_dims' : out_shape,
                 'params' : in_shape * out_shape}
        return layer

    def build_soft(self, layer):

        in_shape = np.product(self.layers[-1]['out_dims'])
        out_shape = np.asarray(layer['n_out'])

        construct = nn.SoftmaxLayer(
            n_in= in_shape,
            n_out = layer['n_out'],
            p_dropout = layer['p_dropout']
        )

        layer = {'construct' : construct,
                 'out_dims' : out_shape,
                 'params' : in_shape * out_shape}
        return layer

    def run(self):
        self.training_data, self.validation_data, self.test_data = load_data_shared(
            self.network['Data']['TRAIN_STACK'], self.network['Data']['TRAIN_LABELS'],
            self.network['Data']['VALID_STACK'], self.network['Data']['VALID_LABELS'],
            self.network['Data']['TEST_STACK'], self.network['Data']['TEST_LABELS'])

        net = nn.Network(self.layers_construct,
                         self.minibatch_size,
                         params_file = self.params_file,
                         logfile=self.logfile,
                         figfile=self.figfile,
                         restart=self.restart)
        net.SGD(self.training_data, self.network['epochs'],
                self.minibatch_size, self.network['eta'],
		self.network['eta_decay'], self.network['eta_interval'],
                self.validation_data, self.test_data,
                self.network['lmbda'],
                self.network['descent_method'],
                self.network['mu'])

    def classify(self, data):
        net = nn.Network(self.layers_construct,
                         self.minibatch_size,
                         params_file = self.params_file,
                         logfile=self.predictions,
                         restart=True)

        predictions = net.classify(data)
        return predictions

    def analyze(self):
        memory = 0
        compute = 0
        for layer in self.layers:
            print(layer['construct'])
            print("Memory Footprint:")
            print(layer['out_dims'])
            layer_memory = np.product(layer['out_dims'])
            print(layer_memory)
            memory += layer_memory
            print("Compute Footprint:")
            print(layer['params'])
            layer_compute = np.product(layer['params'])
            print(layer_compute)
            compute += layer_compute

        print("Memory Footprint: {0}".format(memory))
        print("Compute Footprint: {0}".format(compute))


if __name__ == '__main__':

    network_file = os.path.abspath(sys.argv[1])

    print "started at: " + time.asctime()
    t1_asc = time.asctime()
    t1 = time.time()

    net = build_network(network_file)
    net.run()

    print "started at: " + t1_asc
    print "finished at: " + time.asctime()

    TIME=time.time() - t1
    HOURS = TIME/3600
    MINUTES = (TIME%3600)/60
    SECONDS = (TIME%3600)%60

    print "duration:{0:3.0f} hours, {1:3.0f} minutes, {2:3.0f} seconds".format(
                                                          HOURS,MINUTES,SECONDS)


