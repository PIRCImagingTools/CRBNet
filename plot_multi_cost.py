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
            if row[0] != 'Epoch':
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
                ylab='Mean Epoch Cost (log scale)'):
    fig1=plt.figure(figsize=(10,6))
    fig1.set_facecolor([1,1,1])
    ax=fig1.add_subplot(111)
    for i in xrange(len(data)):
        x=np.arange(0, data[i].shape[0])
        ax.plot(x, data[i][:], lw=3, color=colors[i], alpha=0.8, label=labels[i])
    ax.set_xlim(0, 500)
    #ax.set_ylim(axis[2],axis[3])
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylab)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
             fancybox=True, shadow=True, ncol=3)
    fig1.savefig(sfile, dpi=300, facecolor=fig1.get_facecolor(), edgecolor='w',
                orientation='landscape', bbox_inches=None, pad_inches=0.1)


if __name__ == '__main__':
    parent_dir = '/home/rafa/CRBnet_20170222/'
    save_file = parent_dir+'Tanhcost.png'

    r17 = load_csv(parent_dir+'Run_17/accuracy.csv')
    r22 = load_csv(parent_dir+'Run_22/accuracy.csv')
    r23 = load_csv(parent_dir+'Run_23/accuracy.csv')
    r24 = load_csv(parent_dir+'Run_24/accuracy.csv')
    r25 = load_csv(parent_dir+'Run_25/accuracy.csv')
    r26 = load_csv(parent_dir+'Run_26/accuracy.csv')
    r27 = load_csv(parent_dir+'Run_27/accuracy.csv')



    csvs = [r17, r22, r23, r24, r25, r26, r27]
    labels = ['50/25/15 20v20d', '15/25/50 20v20d',
              '15/25/50 20v20d', '15/25/50 30v30d',
              '15/25/50 40v40d', '50/25/15 40v40d',
              '50/25/15 1v40d']
    data = stack_data(csvs, 4)
    plot_metric(data,labels, title="Mean Epoch Cost", sfile=save_file)

