# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:32:41 2018

@author: FPTShop
"""

import numpy as np

x = np.random.randn(100,2)
y = np.array([[1]*100]).T
w = np.random.randn(3)
data = np.concatenate((x,y),axis=1)
d = data.dot(w)

def sigmoid(d):
    return 1/(1+np.exp(-d))

print(sigmoid(d))

a = np.arange(100)

