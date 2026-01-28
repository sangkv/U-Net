import numpy as np
import random


def random_flip(img, mask):
    if random.random() < 0.5:
        img = np.flip(img, axis=0)
        mask = np.flip(mask, axis=0)
    if random.random() < 0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)
    return img, mask


def random_rot90(img, mask):
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
    return img, mask


def random_intensity_shift(img, max_shift=0.1):
    shift = random.uniform(-max_shift, max_shift)
    img = img + shift
    return img


def random_gaussian_noise(img, sigma=0.02):
    if random.random() < 0.5:
        noise = np.random.normal(0, sigma, img.shape)
        img = img + noise
    return img


import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic_deform(img, mask, alpha=20, sigma=4):
    """
    alpha: strain strength (EM should be small)
    Sigma: smoothness (the higher the value, the smoother)
    """

    shape = img.shape

    dx = gaussian_filter(
        (np.random.rand(*shape) * 2 - 1),
        sigma
    ) * alpha

    dy = gaussian_filter(
        (np.random.rand(*shape) * 2 - 1),
        sigma
    ) * alpha

    x, y = np.meshgrid(
        np.arange(shape[1]),
        np.arange(shape[0])
    )

    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1))
    )

    img_deformed = map_coordinates(
        img, indices, order=1, mode='reflect'
    ).reshape(shape)

    mask_deformed = map_coordinates(
        mask, indices, order=0, mode='reflect'
    ).reshape(shape)

    return img_deformed, mask_deformed


def em_augmentation(img, mask):
    img, mask = random_flip(img, mask)
    img, mask = random_rot90(img, mask)
    img = random_intensity_shift(img)
    img = random_gaussian_noise(img)
    if random.random() < 0.3:
        img, mask = elastic_deform(img, mask, alpha=10, sigma=5)
    return img, mask
