# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:33:40 2021

@author: von.gostev
"""
import __init__
from pyMMF import IndexProfile
import matplotlib.pyplot as plt

area_size = 3.5 * 31.25

profile = IndexProfile(npoints=512, areaSize=area_size)
profile.initStepIndexMultiCoreRadial(central_core_radius=0, a=6 / 2, core_pitch=35., dims=12, NA=0.19,
                                     n1=1.4613)

fig = plt.figure(figsize=(5, 4), dpi=200)
plt.imshow(profile.n, extent=[-area_size / 2, area_size / 2] * 2)
plt.xlabel('x, мкм.')
plt.ylabel('y, мкм.')
plt.colorbar()
plt.tight_layout()
plt.savefig('simc/simc_index.png', dpi=200)
plt.show()
