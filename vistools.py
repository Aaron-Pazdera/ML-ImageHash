import numpy as np
import matplotlib.pyplot as plt
from skimage.io._plugins.pil_plugin import pil_to_ndarray
from torchvision.transforms import ToPILImage


class ShowTensor():
    def __call__(self, tensor):
        npimg = tensor.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        plt.show()


show_tensor = ShowTensor()


class ShowTriplet():
    def __call__(self, triplet, axes=True):
        npimgs = [tensor.numpy() for tensor in triplet]
        axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 12))[1]
        axs[0].imshow(np.transpose(npimgs[0], (1, 2, 0)),
                      interpolation='nearest')
        axs[1].imshow(np.transpose(npimgs[1], (1, 2, 0)),
                      interpolation='nearest')
        axs[2].imshow(np.transpose(npimgs[2], (1, 2, 0)),
                      interpolation='nearest')
        axs[0].set_title("Anchor (A)")
        axs[1].set_title("Positive (P)")
        axs[2].set_title("Negative (N)")
        if not axes:
            axs[0].set_axis_off()
            axs[1].set_axis_off()
            axs[2].set_axis_off()
        plt.show()


show_triplet = ShowTriplet()


def show_ntuples(ntuple, axes=True):
    ntuple = ntuple[0] + ntuple[1]
    n = len(ntuple)
    n2 = n//2

    pt = ToPILImage()
    npimgs = [pt(tensor) for tensor in ntuple]

    axs = plt.subplots(2, n2, figsize=(16, 5))[1]
    axs = np.ravel(axs)

    for i in range(n):
        axs[i].imshow(npimgs[i])
        if not axes:
            axs[i].set_axis_off()

    plt.show()
