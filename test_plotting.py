import matplotlib.pyplot as plt
import numpy as np
import time

def plot_metrics(figfile, epoch, cost, train, val, test):
    fig1 = plt.figure(figsize=(10,6))
    fig1.set_facecolor([1,1,1])
    ax1=fig1.add_subplot(211)
    x = np.arange(0, epoch)
    ax1.plot(x, cost, label="Mean Training Cost")
    ax1.set_title("Mean Cost", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Mean Cost")
    ax1.set_yscale('log')
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0 + box1.height * 0.2,
                      box1.width, box1.height * 0.9])
#    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#               fancybox=True, shadow=True, ncol=3)

    ax2 = fig1.add_subplot(212)
    ax2.plot(x, train, color='blue', alpha=0.8, label="Training Accuracy")
    ax2.plot(x, val, color='green', alpha=0.8, label="Validation Accuracy")
    ax2.plot(x, test, color='red', alpha=0.8, label="Test Accuracy")
    ax2.set_title("Accuracy", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0 + box2.height * 0.2,
                      box2.width, box2.height * 0.9])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
               fancybox=True, shadow=True, ncol=3)


    fig1.savefig(figfile, dpi=300, facecolor=fig1.get_facecolor(),
                 edgecolor='w', orientation='landscape',
                 bbox_inches=None, pad_inches=0.1)
    plt.close()


if __name__ == '__main__':

    epochs = 200
    figfile = '/home/rafa/Desktop/fake_graph.png'

    fake_cost = [1,]
    fake_train = [0.5,]
    fake_val = [0.5,]
    fake_test = [0.5,]
    for i in xrange(epochs):
        fake_epoch = i+1
        print fake_epoch
        plot_metrics(figfile, fake_epoch, fake_cost,
                     fake_train, fake_val, fake_test)

        time.sleep(0.5)
        fake_cost.append(fake_cost[-1]/1.5)
        fake_train.append(fake_train[-1]*1.02)
        fake_val.append(fake_val[-1]*1.005)
        fake_test.append(fake_test[-1]*1.002)



