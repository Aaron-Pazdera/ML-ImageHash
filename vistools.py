import numpy as np
import matplotlib.pyplot as plt
from skimage.io._plugins.pil_plugin import pil_to_ndarray
from torchvision.transforms import ToPILImage
from PIL import Image


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


def show_imgpair(pair, axes=True):
    pt = ToPILImage()
    npimgs = (pt(pair[0]), pt(pair[1]))

    axs = plt.subplots(1, 2, figsize=(10, 5))[1]
    axs = np.ravel(axs)
    axs[0].imshow(npimgs[0])
    axs[1].imshow(npimgs[1])
    if not axes:
        axs[0].set_axis_off()
        axs[1].set_axis_off()
    plt.show()


def show_matches(matches, axes=True):
    print(matches)
    fname_list = [tup[1] for tup in matches]
    imgs = [Image.open(fname) for fname in fname_list]

    axs = plt.subplots(1, len(imgs), figsize=(10, 5))[1]
    axs = np.ravel(axs)

    for i in range(len(imgs)):
        axs[i].imshow(imgs[i])
        if not axes:
            axs[i].set_axis_off()
    plt.show()
