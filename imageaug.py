from torchvision.transforms.functional import rotate
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, RandomGrayscale, ToTensor, ToPILImage
import random
import numpy as np
from skimage.util import random_noise
from skimage.io._plugins.pil_plugin import pil_to_ndarray, ndarray_to_pil


class ApplyOne:
    def __init__(self, *argv):
        if len(argv) == 0:
            raise AssertionError(
                'Must take at least one transform as arguments.')
        self.transforms = argv

    def __call__(self, img):
        return random.choice(self.transforms)(img)


class RandomRotate:
    # Rotate by one of the given angles.
    def __init__(self, angles):
        self.angles = angles
        self.flip = RandomHorizontalFlip(.5)

    def __call__(self, x):
        angle = random.choice(self.angles)
        return rotate(self.flip(x), angle)


class RandomFillCrop:
    def __init__(self, chance):
        self.chance = chance
        self.rcs = [RandomCrop((64, 64), padding=4, fill=0),
                    RandomCrop((64, 64), padding=4, fill=255)]

    def __call__(self, img):
        return random.choice(self.rcs)(img) if random.random() < self.chance else img


class Greyscale:
    def __init__(self):
        pass

    def __call__(self, img):
        return img.convert('L')


class Noise:
    def __init__(self):
        self.to_PIL = ToPILImage('RGB')

    def __call__(self, img):
        noise_type = random.random()
        if noise_type < .33:
            noise_type = 's&p'
        elif noise_type < .66:
            noise_type = 'speckle'
        elif noise_type < 1.:
            noise_type = 'gaussian'
        else:
            return img

        nd_img = np.array(img) / 255.0
        nd_noised = random_noise(np.asarray(
            nd_img), mode=noise_type, clip=True)
        img = ndarray_to_pil(nd_noised)
        return img


class ToRGBTensor:
    # Used in model
    def __init__(self):
        self.tt = ToTensor()

    def __call__(self, img):
        return self.tt(img.convert('RGB'))
