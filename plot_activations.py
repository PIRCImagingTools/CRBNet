import cPickle
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from scipy.interpolate import griddata
from scipy.misc import imresize

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
    elif fig == 'cam':
        colormap=plt.cm.jet

    shape = activations.shape
    if len(shape) < 2:
        return 0
    filters = shape[0] #10
    image_size = shape[1:] #[48,43,33]
    slices = shape[-1]

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


def plot_cams(cams, figfile):
    colormap=plt.cm.jet

    shape = cams.shape
    if len(shape) < 2:
        return 0
    image_size = shape #[48,43,33]
    slices = shape[-1]

    fig = plt.figure(figsize=(6, slices*3))
    fig.set_facecolor([1,1,1])

    k = 1

    for j in xrange(slices):
        image_slice = cams[:,:,j].squeeze()
        ax = fig.add_subplot(slices, 2,  k)
        ax.set_title('Slice: {0} Native'.format(j))
        plt.imshow(image_slice, cmap=colormap, interpolation='none')
        plt.axis("off")
        k+=1
        ax = fig.add_subplot(slices,2,  k)
        ax.set_title('Slice: {0} Interpolated'.format(j))
        plt.imshow(image_slice, cmap=colormap, interpolation='spline16')
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

    #Original dimensions to project CAMs onto
    oX = 100
    oY = 90
    oZ = 70

    #one for each conv layer
    for x in xrange(len(activations)):
        activationfile = os.path.join(main_dir, 'layer_{0}_activations.png'.format(x))
        activationfile_0 = os.path.join(main_dir, 'layer_{0}_activations_class0.png'.format(x))
        activationfile_1 = os.path.join(main_dir, 'layer_{0}_activations_class1.png'.format(x))
        camfile_0 = os.path.join(main_dir, 'layer_{0}_cam_0.png'.format(x))
        camfile_1 = os.path.join(main_dir, 'layer_{0}_cam_1.png'.format(x))

        difffile = os.path.join(main_dir, 'layer_{0}_diff_activations.png'.format(x))

        #Entire layer - All feature maps for each subject
        # [X, Y, Z, Z, Z] where X is each subject, Y is each feature map, Z is
        # dimensions
        layer = np.asarray(activations[x]).squeeze()

        #Separate each class
        layer_class0 = layer[np.where(classifications == 0)]
        layer_class1 = layer[np.where(classifications == 1)]

        # Mean activations (across subjects) for each feature map
        mean_0 = np.mean(layer_class0, axis=0)
        mean_1 = np.mean(layer_class1, axis=0)


        #Class activation maps (mean across feature maps) broadcast to retain 4D
        cam_0 = np.mean(mean_0, axis=0)

        #Class activation maps (mean across feature maps) broadcast to retain 4D
        cam_1 = np.mean(mean_1, axis=0)


        diff_layer = mean_0 - mean_1
        mean_layer = np.mean(layer, axis=0)

        plot_activations(mean_layer, activationfile, fig='activation')
        plot_activations(diff_layer, difffile, fig='diff')

        plot_activations(mean_0, activationfile_0, fig='activation')
        plot_activations(mean_1, activationfile_1, fig='activation')

        plot_cams(cam_0, camfile_0)
        plot_cams(cam_1, camfile_1)




