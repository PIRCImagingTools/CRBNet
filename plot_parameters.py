import cPickle
import numpy as np
import matplotlib.pyplot as plt
import os,sys

def load_parameters(params_file):
    params_list = []
    f = open(params_file, 'rb')
    parameters = cPickle.load(f)
    for x in xrange(len(parameters)):
        W,b = parameters[x]
        if W.name == 'conv_w':
            params_list.append(np.asarray(W.get_value()))

    return  params_list


def plot_parameters(params, figfile):
    shape = params.shape
    filters = shape[0] #10
    image_size = shape[2:] #[4,4,4]
    slices = shape[4]

    fig = plt.figure(figsize=(filters*3, slices*3))
    fig.set_facecolor([1,1,1])

    k = 1

    for j in xrange(slices):
        for i in xrange(filters):
            image_slice = params[i,0,:,:,j].squeeze()
            ax = fig.add_subplot(slices, filters, k)
            ax.set_title('Filter: {0}, Slice: {1}'.format(i,j))
            plt.imshow(image_slice, cmap=plt.cm.gray, clim=(-1.0,1.0), interpolation='none')
            plt.axis("off")
            k += 1
    fig.savefig(figfile, dpi=300, facecolor=fig.get_facecolor(),
                 edgecolor='w', orientation='landscape',
                 bbox_inches=None, pad_inches=0.1)
    plt.close()



if __name__ == '__main__':
    main_dir = os.path.abspath(sys.argv[1])

    params_file = os.path.join(main_dir, 'params.save')
    params = load_parameters(params_file)
    for x in xrange(len(params)):
        figfile = os.path.join(main_dir, 'layer_{0}_weights_norm.png'.format(x))
        plot_parameters(params[x], figfile)



