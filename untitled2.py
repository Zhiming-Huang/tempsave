# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:26:05 2017

@author: CNLAB
"""
from FCM import *

a=FCM(dataSet,2)

centroids=a.randCent(3).A

a.row, a.col=a.dataset.shape
u2,centroids=a.alg_Pfcm(centroids)