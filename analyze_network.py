import numpy as np
from nipy import load_image
import os,sys
from run_network import build_network


if __name__ == '__main__':
    network_file = os.path.abspath(sys.argv[1])

    net = build_network(network_file, classify=True)
    analysis = net.analyze()
    print(analysis)

