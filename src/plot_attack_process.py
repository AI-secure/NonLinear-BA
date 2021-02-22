import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PGENs = ['naive', 'resize9408', 'DCT9408', 'PCA9408basis']
PGEN_NAMEs = ['HSJA', 'QEBA-S', 'QEBA-F', 'QEBA-I']
src_image = np.load('steps/imagenet__src.npy')
tgt_image = np.load('steps/imagenet__tgt.npy')

# Process
if True:

    STEPS = (1,5,10,20)
    N = len(STEPS)
    fig = plt.figure(figsize=(9,6))
    plt.subplot(4,N+1,1)
    plt.subplots_adjust(wspace=0.0,hspace=0.20)
    plt.imshow(src_image)
    plt.xlabel('Source Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(4,N+1,3*(N+1)+1)
    plt.imshow(tgt_image)
    plt.xlabel('Target Image')
    plt.xticks([])
    plt.yticks([])
    for pid, PGEN in enumerate(PGENs):
        for i,step in enumerate(STEPS):
            #plt.subplot(4,N+1,pid*(N+1)+i+2)
            ax = fig.add_subplot(4,N+1,pid*(N+1)+i+2)
            data = np.load('steps/perturbed%s%d.npz'%(PGEN,step))
            plt.imshow(data['pert'])
            #plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            if i == N-1:
                #ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")

                plt.ylabel(PGEN_NAMEs[pid])
                #plt.tick_right()
            if (pid == 3):
                plt.xlabel('d=%.2e\n#q=%d'%(data['info'][1], data['info'][0]))
            else:
                plt.xlabel('d=%.2e'%data['info'][1])
        #plt.subplot(4,12,pid*12+12)
        #plt.imshow(np.abs(data['pert']-tgt_image))
        #plt.xticks([])
        #plt.yticks([])
    plt.show()
    fig.savefig('process.pdf', bbox_inches='tight')

# Diff
if False:
    fig = plt.figure(figsize=(20,5))
    for pid, PGEN in enumerate(PGENs):
        plt.subplot(1,4,pid+1)
        data = np.load('steps/perturbed%s99.npz'%(PGEN))
        plt.imshow(5*np.abs(data['pert']-tgt_image))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    fig.savefig('diff.pdf')