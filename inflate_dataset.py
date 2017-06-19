import getpass
from scipy.ndimage.interpolation import affine_transform
from numpy import random
import numpy as np
from nipy import load_image, save_image
from nipy.core.api import Image

user=getpass.getuser()

def transrotate(img):

    offset = random.randint(-3,3, size=3)
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

    samples_per_subject = 5
    merged_stack  = "./res/CHP_Stacked_HP_CROP_BIN_20170501.nii.gz"
    stack_label = "./res/CHP_HP_Labels_20170501.txt"
    labels_out_file = "./res/HP_Inflated_Train_20170501_Labels.txt"
    params_out_file = "./res/HP_Inflated_Train_20170501_6v10d_Params.csv"
    out_file = "./res/HP_Inflated_Train_20170501.nii.gz"

    with open(stack_label) as stack_labs:
        labels = stack_labs.read().splitlines()

    nii_stack = load_image(merged_stack)
    coord = nii_stack.coordmap
    data_stack = nii_stack.get_data()

    data_out = data_stack
    print(data_out.shape)

    labels_out = open(labels_out_file, 'w')
    for line in labels:
        labels_out.writelines(line+'\n')

    params_file = open(params_out_file, 'w')
    params_file.write('XT,YT,ZT,DEG\n')

    for subj in xrange(data_out.shape[3]):
        #only inflate dysplastic ones
        if labels[subj] == '1':
            for i in xrange(samples_per_subject):
                brain_A = np.expand_dims(data_stack[:,:,:,subj], axis=3)
                params, new_data = transrotate(brain_A)
                params_file.write("{0},{1},{2},{3}\n".format(*params))
                data_out = np.append(data_out, new_data, axis=3)
                labels_out.write('1\n')

    ### last one to even out dataset
    brain_A = np.expand_dims(data_stack[:,:,:,0], axis=3)
    params, new_data = transrotate(brain_A)
    params_file.write("{0},{1},{2},{3}\n".format(*params))
    data_out = np.append(data_out, new_data, axis=3)
    labels_out.write('1\n')



    print("Final data size: {0}".format(data_out.shape))
    arr_img = Image(data_out, coord)
    save_image(arr_img, out_file)

    labels_out.close()
    params_file.close()
