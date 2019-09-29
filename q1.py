#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:20:11 2019

@author: lokeshkumar
"""

import cv2
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

image = cv2.imread('Lena.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)/255
plt.imshow(gray,cmap='gray')
plt.imsave('1Lena.png',gray,cmap='gray')

im = Image.open('1Lena.png')

half = Image.new('RGB',tuple([ int(d/2) for d in im.size ]) )
print(half.size)

for i in range( im.size[0] ):
    for j in range( im.size[1] ):
        if i % 2 == 0 and j % 2 == 0:
            half.putpixel(( int(i/2), int(j/2) ), im.getpixel((i,j)) )

#half.show()
half.save('1aLena.png')

im1 = Image.open('1aLena.png')

half1 = Image.new('RGB',tuple([ int(d/2) for d in im1.size ]) )
print(half1.size)
#half = Image.new('RGB', tuple([ int(d/4) for d in im.size ]))

for i in range( im1.size[0] ):
    for j in range( im1.size[1] ):
        if i % 2 == 0 and j % 2 == 0:
            half1.putpixel(( int(i/2), int(j/2) ), im1.getpixel((i,j)) )

#half1.show()
half1.save('1bLena.png')

im2 = Image.open('1bLena.png')

half2 = Image.new('RGB',tuple([ int(d*2) for d in im2.size ]) )
print(half2.size)

for i in range( im2.size[0] ):
    for j in range( im2.size[1] ):
        #if i % 2 == 0 and j % 2 == 0:
        half2.putpixel(( (2*i), (2*j) ), im2.getpixel((i,j)) )
        half2.putpixel(( (2*i+1), (2*j+1) ), im2.getpixel((i,j)) )

#half.show()
half2.save('1cLena.png')

