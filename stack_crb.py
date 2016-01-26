from nipy import load_image, save_image
import numpy as np
import getpass
from nipy.core.apy import Image

user=getpass.getuser()

RCRB=18
LCRB=17

def get_crb(segmentation, output):
    brain = load_image(segmentation)


