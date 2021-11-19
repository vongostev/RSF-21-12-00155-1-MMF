# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 00:15:45 2021

@author: von.gostev
"""

import __init__
from pyMMF import IndexProfile
import matplotlib.pyplot as plt

area_size = 150

profile = IndexProfile(npoints=1024, areaSize=area_size)
# # HC-2000-01
# profile.initPhotonicCrystalHex(central_core_radius=7.5, central_core_n1=1, 
#                                a=2.4, core_pitch=0., NA=0.2, pcf_radius=44,
#                                n1=1.4613, cladding_radius=155 / 2)
# HC-1550-01
profile.initPhotonicCrystalHex(central_core_radius=5, central_core_n1=1, 
                               a=1.9, core_pitch=0., NA=0.2, pcf_radius=35.3,
                               n1=1.4613, cladding_radius=60)

fig = plt.figure(figsize=(5, 4), dpi=200)
plt.imshow(profile.n, extent=[-area_size / 2, area_size / 2] * 2)
plt.xlabel('x, мкм.')
plt.ylabel('y, мкм.')
plt.colorbar()
plt.tight_layout()
plt.savefig('pcf/pcf_index.png', dpi=200)
plt.show()
