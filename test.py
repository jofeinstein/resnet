import matplotlib.pyplot as plt
import numpy as np

def list_plot(lst,title):
    fig = plt.figure()
    for i in range(len(lst)):
        plt.plot(lst[i])
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.draw()
    fig.savefig('./log/' + title + '.png', dpi=fig.dpi)
    plt.show()

lst = [[1,5,2],[5,9,0,1,2]]
list_plot(lst,'test')