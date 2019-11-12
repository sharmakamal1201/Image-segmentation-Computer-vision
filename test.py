#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:07:30 2019

@author: ramanathan
"""

import numpy as np
import matplotlib.pyplot as plt
import os
#import cv2
from os.path import join
from matplotlib import cm
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import LightSource
import matplotlib.image as mpimg

plt.close('all')
clear = lambda: os.system('clear')
clear()

img = mpimg.imread(join(''.join(['Input/tiger.png'])));

plt.imshow(img)