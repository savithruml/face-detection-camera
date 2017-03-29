#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: SAVITHRU M LOKANATH

import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('rhosp.jpg', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()