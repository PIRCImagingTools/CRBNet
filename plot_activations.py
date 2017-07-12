import cPickle
import numpy as np
import matplotlib.pyplot as plt
import os,sys

def load_text_data(text_file):
    predictions = np.loadtxt(open(text_file, 'rb'), delimiter=',',skiprows=1)
    classification = predictions[:,2]
    return classification

def load_activations(activations_file):
    f = open(activations_file, 'rb')
    activations = cPickle.load(f)
    return activations


def plot_activations(activations, figfile, fig):
    if fig == 'activation':
        colormap = plt.cm.gray
    elif fig == 'diff':
        colormap = plt.cm.seismic

    shape = activations.shape
    if len(shape) < 2:
        return 0
    filters = shape[0] #10
    image_size = shape[1:] #[48,43,33]
    slices = shape[3]

    fig = plt.figure(figsize=(filters*3, slices*3))
    fig.set_facecolor([1,1,1])

    k = 1

    for j in xrange(slices):
        for i in xrange(filters):
            image_slice = activations[i,:,:,j].squeeze()
            ax = fig.add_subplot(slices, filters, k)
            ax.set_title('Filter: {0}, Slice: {1}'.format(i,j))
            plt.imshow(image_slice, cmap=colormap, clim=(-1.0,1.0),
                       interpolation='none')
            plt.axis("off")
            k += 1
    fig.savefig(figfile, dpi=300, facecolor=fig.get_facecolor(),
                 edgecolor='w', orientation='landscape',
                 bbox_inches=None, pad_inches=0.1)
    plt.close()



if __name__ == '__main__':
    main_dir = os.path.abspath(sys.argv[1])

    activation_file = os.path.join(main_dir, 'activations.save')
    classification_file = os.path.join(main_dir, 'predictions.csv')

    activations = load_activations(activation_file)
    classifications = load_text_data(classification_file)

    for x in xrange(len(activations)):
        activationfile = os.path.join(main_dir, 'layer_{0}_activations.png'.format(x))
        activationfile_0 = os.path.join(main_dir, 'layer_{0}_activations_class0.png'.format(x))
        activationfile_1 = os.path.join(main_dir, 'layer_{0}_activations_class1.png'.format(x))

        difffile = os.path.join(main_dir, 'layer_{0}_diff_activations.png'.format(x))

        layer = np.asarray(activations[x]).squeeze()
        layer_class0 = layer[np.where(classifications == 0)]
        layer_class1 = layer[np.where(classifications == 1)]

        mean_0 = np.mean(layer_class0, axis=0)
        mean_1 = np.mean(layer_class1, axis=0)

        diff_layer = mean_0 - mean_1
        mean_layer = np.mean(layer, axis=0)

        plot_activations(mean_layer, activationfile, fig='activation')
        plot_activations(mean_0, activationfile_0, fig='activation')
        plot_activations(mean_1, activationfile_1, fig='activation')

        plot_activations(diff_layer, difffile, fig='diff')



