import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

#iWantHue selection of 20 distinct colors

colors = ["#bd8a79", "#7a3fcf", "#73d350",
          "#cf52c3", "#ccd049", "#522a7e",
          "#67d7a6", "#d84c3c", "#7778d6",
          "#c88938", "#556891", "#c8d099",
          "#51284a", "#5d7c37", "#c54a7c",
          "#52897d", "#7c3729", "#88c3d9",
          "#3b3c2d", "#cd9fca"]

def load_csv(csv_file):
    data = []
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        headers = reader.next()
        for row in reader:
            data.append(row)
    return np.asarray(data)

def stack_data(csv_list, idx):
    data = []
    for i in xrange(len(csv_list)):
        data.append(csv_list[i][:,idx])
    return data

def plot_metric(data,labels,
                title="Mean Cost",
                sfile='/home/rafa/Desktop/plot',
                ylab='Validation Accuracy'):
    fig1=plt.figure(figsize=(10,6))
    fig1.set_facecolor([1,1,1])
    ax=fig1.add_subplot(111)
    for i in xrange(len(data)):
        x=np.arange(0, data[i].shape[0])
        ax.plot(x, data[i][:], lw=3, color=colors[i], alpha=0.8, label=labels[i])
    ax.set_xlim(0, 100)
    ax.set_ylim(0.4,1.1)
    #ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylab)
    ax.legend()
    fig1.savefig(sfile, dpi=300, facecolor=fig1.get_facecolor(), edgecolor='w',
                orientation='landscape', bbox_inches=None, pad_inches=0.1)


if __name__ == '__main__':
    parent_dir = '/home/rafa/Dropbox/ws_Python/NN/CRBnet/'
    save_file = parent_dir+'Accuracy.png'

    r11 = load_csv(parent_dir+'Run_11/accuracy_1.csv')
    r12 = load_csv(parent_dir+'Run_12/accuracy.csv')
    r13 = load_csv(parent_dir+'Run_13/accuracy.csv')
    r14 = load_csv(parent_dir+'Run_14/accuracy.csv')
    r15 = load_csv(parent_dir+'Run_15/accuracy.csv')

    csvs = [r11, r12, r13, r14, r15]
    labels = ['1 vox/1 deg', '5 vox/1 deg', '5 vox/5 deg', '10 vox/10 deg', '5-10 vox/ 5-10 deg']
    data = stack_data(csvs, 2)
    plot_metric(data,labels, title="Validation Accuracy", sfile=save_file)

