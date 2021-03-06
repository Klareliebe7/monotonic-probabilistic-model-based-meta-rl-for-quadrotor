import matplotlib.pyplot as plt
import numpy as np
def plot_learning(x, scores,scores_error , filename, lines=None):
    fig=plt.figure(figsize=(18,6), dpi=300)
    # ax=fig.add_subplot(111, label="1"  )
    ax2=fig.add_subplot(111, label="2", frame_on=False  )

    # ax.plot(x, epsilons, color="C0",label = "epsilon")
    ax2.set_xlabel("Simulations" )
    #ax2.set_ylabel("Epsilon", color="C0")
    # ax.tick_params(axis='x', colors="C0")
    # ax.tick_params(axis='y', colors="C0")
    ax2.errorbar(x, scores, yerr=scores_error, fmt='-o',color="C1",label = "Eva. reward",elinewidth = 1,ms = 5,mfc="wheat",mec="salmon",capsize = 3)
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(True)
    #ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Average reward' )
    #ax2.xaxis.set_label_position('top')
    #ax2.yaxis.set_label_position('right')？
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")
    #ax.legend()
    ax2.legend()
    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)