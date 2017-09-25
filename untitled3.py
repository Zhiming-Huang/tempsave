# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:22:01 2017

@author: CNLAB
"""

import matplotlib.pyplot as plt
from matplotlib import cm  
import numpy as np

fig, axes = plt.subplots(2,3,figsize=(12, 12))
cmap=cm.get_cmap('rainbow',1000) 
#axes[0].set_title('Iteration 1 of Former FCM')
for x_val in range(3):
    for y_val in range(7):
        c=f1[y_val][x_val]
        axes[0][0].text(x_val, y_val, c, va='center', ha='center')

for x_val in range(3):
    for y_val in range(7):
        c=f2[y_val][x_val]
        axes[0][1].text(x_val, y_val, c, va='center', ha='center')

for x_val in range(3):
    for y_val in range(7):
        c=f3[y_val][x_val]
        axes[0][2].text(x_val, y_val, c, va='center', ha='center')

axes[0][0].imshow(f1,interpolation='nearest',cmap=cmap,aspect='auto')
axes[0][0].set_xticks([0,1,2])
axes[0][0].set_title("Iteration 1 of Former FCM")
axes[0][0].set_xlabel("Cluster")
axes[0][0].set_ylabel("MPCs")
axes[0][1].imshow(f2,interpolation='nearest',cmap=cmap,aspect='auto')
axes[0][1].set_xticks([0,1,2])
axes[0][1].set_title("Iteration 2 of Former FCM")
axes[0][1].set_xlabel("Cluster")
axes[0][1].set_ylabel("MPCs")
axes[0][2].imshow(f3,interpolation='nearest',cmap=cmap,aspect='auto')
axes[0][2].set_xticks([0,1,2])
axes[0][2].set_title("Iteration 6 of Former FCM")
axes[0][2].set_xlabel("Cluster")
axes[0][2].set_ylabel("MPCs")

for x_val in range(3):
    for y_val in range(7):
        c=n1[y_val][x_val]
        axes[1][0].text(x_val, y_val, c, va='center', ha='center')

for x_val in range(3):
    for y_val in range(7):
        c=n2[y_val][x_val]
        axes[1][1].text(x_val, y_val, c, va='center', ha='center')

for x_val in range(3):
    for y_val in range(7):
        c=n3[y_val][x_val]
        axes[1][2].text(x_val, y_val, c, va='center', ha='center')

axes[1][0].imshow(n1,interpolation='nearest',cmap=cmap,aspect='auto')
axes[1][0].set_xticks([0,1,2])
axes[1][0].set_title("Iteration 1 of Modified FCM")
axes[1][0].set_xlabel("Cluster")
axes[1][0].set_ylabel("MPCs")
axes[1][1].imshow(n2,interpolation='nearest',cmap=cmap,aspect='auto')
axes[1][1].set_xticks([0,1,2])
axes[1][1].set_title("Iteration 2 of Modified FCM")
axes[1][1].set_xlabel("Cluster")
axes[1][1].set_ylabel("MPCs")
axes[1][2].imshow(n3,interpolation='nearest',cmap=cmap,aspect='auto')
axes[1][2].set_xticks([0,1,2])
axes[1][2].set_title("Iteration 6 of Modified FCM")
axes[1][2].set_xlabel("Cluster")
axes[1][2].set_ylabel("MPCs")
plt.grid(False)

