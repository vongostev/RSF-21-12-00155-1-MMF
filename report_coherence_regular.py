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

profile = IndexProfile(npoints=512, areaSize=3.5 * 25)
profile.initPhotonicCrystalHex(central_core_radius=7 / 2, a=2.3 / 2, core_pitch=0.,
                               pcf_radius=24,  cladding_radius=65)

plt.imshow(profile.n)
