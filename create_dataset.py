import getpass
from scipy.ndimage.interpolation import affine_transform
from numpy import random
import numpy as np
from nipy import load_image, save_image
from nipy.core.api import Image

user=getpass.getuser()

samples_per_subject = 10
merged_stack  = "./res/MERGED_CRBS_crop.nii.gz"
labels_file = "./res/synth_labels_test_10v10d.txt"
params_file = "./res/synth_params_test_10v10d.csv"
out_file = "./res/synth_data_test_10v10d.nii.gz"


def transrotate(img):

    offset = random.randint(-10,10, size=3)
    angle = random.randint(11)
    angle_r = angle * np.pi/180
    c = np.cos(angle_r)
    s = np.sin(angle_r)

    print("Offset: {0}".format(offset))
    print("Angle: {0}, ({1} radians)".format(angle, angle_r))
    img3 = np.squeeze(img)
    matrix = np.asarray([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])

    params = np.append(offset,angle)
    return params, np.where(np.expand_dims(
             affine_transform(img3, matrix, offset, order=1),
                          axis=3) > 0, 1, 0)

if __name__ == '__main__':

    nii_stack = load_image(merged_stack)
    coord = nii_stack.coordmap
    data_stack = nii_stack.get_data()

    brain_A = np.expand_dims(data_stack[:,:,:,1], axis=3)
    brain_B = np.expand_dims(data_stack[:,:,:,3], axis=3)

    data_out = np.append(brain_A, brain_B, axis=3)
    print(data_out.shape)

    labels_file = open(labels_file, 'w')
    labels_file.write('0\n1\n')

    params_file = open(params_file, 'w')
    params_file.write('XT,YT,ZT,DEG\n0,0,0,0\n0,0,0,0\n')

    for i in xrange(samples_per_subject):
        params, new_data = transrotate(brain_A)
        params_file.write("{0},{1},{2},{3}\n".format(*params))
        data_out = np.append(data_out, new_data, axis=3)
        labels_file.write('0\n')

    for i in xrange(samples_per_subject):
        params, new_data = transrotate(brain_B)
        params_file.write("{0},{1},{2},{3}\n".format(*params))
        data_out = np.append(data_out, new_data, axis=3)
        labels_file.write('1\n')

    print("Final data size: {0}".format(data_out.shape))
    arr_img = Image(data_out, coord)
    save_image(arr_img, out_file)

    labels_file.close()
    params_file.close()
