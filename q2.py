#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:09:00 2019

@author: lokeshkumar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from c_vision.convolution import convolution
 
 
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
 
def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()
 
    return kernel_2D
 
 
def myGaussianSmoothing(image, kernel_size, sigma):
    verbose = True;
    kernel = gaussian_kernel(kernel_size, sigma=sigma, verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)
 
 
if __name__ == '__main__':
    
    image = cv2.imread('1Lena.png')
    img1 = myGaussianSmoothing(image, 3, 1)
plt.imsave('xyz.png',img1,cmap='gray')



"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('1Lena.png')
#cv2.imshow('gg.png',image)
noised = (image + 0.2*np.random.rand(*image.shape).astype(np.float32))/255
noised = noised.clip(0,1)
plt.imshow(noised[:, :, [2,1,0]])
plt.show()

gauss_blur = cv2.GaussianBlur(noised, (11,11),7)
plt.imshow(gauss_blur[:, :, [2,1,0]])

plt.show() 

plt.imsave('2b7Lena.png',gauss_blur)
"""