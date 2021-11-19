# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:33:40 2021

@author: von.gostev
"""
import __init__
from lightprop2d import Beam2D, random_round_hole
from gi import ImgEmulator
from pyMMF import IndexProfile, propagationModeSolver, estimateNumModesSI
import matplotlib.pyplot as plt

profile = IndexProfile(npoints=256, areaSize=3.5 * 31.25)
profile.initStepIndexMultiCoreRadial(central_core_radius=0, a=6 / 2, core_pitch=35., dims=12)

plt.imshow(profile.n)
