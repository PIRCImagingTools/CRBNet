import cPickle
import numpy as np
from nipy import load_image
import theano
from theano import tensor as T
from run_network import build_network


def load_params(param_file):
    with open(param_file, 'rb') as f:
        params = cPickle.load(f)

def load_data(stack, labels):
    def get_nii_data(nifti_file):
        image = load_image(nifti_file)
        data = image.get_data().transpose(3, 0, 1, 2)
        print("data shape:")
        print(data.shape)
        return data

    def get_text_data(labels):
        vector = []
        with open(labels) as f:
            for line in f:
                vector.append(line.rstrip())
        return np.asarray(vector)

    def shared(DATA, LABELS):
        shared_x = theano.shared(
            np.asarray(DATA, dtype=theano.config.floatX),
            borrow=True)
        shared_y = theano.shared(
            np.asarray(LABELS, dtype=theano.config.floatX),
            borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return shared(get_nii_data(stack), get_text_data(labels))


if __name__ == '__main__':
    params_file = './Run_16-C/params.save'
    network_file = './Run_16-C/network.json'
    stack_file = './res/Stacked_CRB_CROP_BIN_Test_20170324.nii.gz'
    labels_file = './res/CRB_Labels_Test_20170324.txt'

    params = load_params(params_file)
    data = load_data(stack_file, labels_file)
    net = build_network(network_file, classify=True)
    net.calc_activations(data)


