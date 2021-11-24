# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:14:30 2021

@author: von.gostev
"""

import __init__
import numpy as np

from lightprop2d import um, nm, cm, mm
from gi.slm import slm_modprofile, dmd_modprofile
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


def mask_plot(mask, ax=plt):
    return ax.imshow(mask,
              extent=[-area_size / 2 / mm, area_size / 2 / mm]*2,
              cmap='jet')


npoints = 2 ** 8
area_size = 0.5*cm


modprofile = slm_modprofile(area_size / npoints,
                        2 * np.pi * np.random.random(size=(50, 50)),
                        pixel_size=100e-4, pixel_gap=0, angle=0)
# modprofile = dmd_modprofile(area_size / npoints,
#                             np.random.random(size=(50, 50)),
#                             pixel_size=100e-4, pixel_gap=0, angle=0)

# Is.append(ibeam.iprofile.get()[npoints // 2 - 6])
fig, ax = plt.subplots(2, 1, figsize=(5, 8), dpi=200)
im = mask_plot(np.abs(modprofile).astype(np.float16).astype(np.float32), ax[0])
fig.colorbar(im, ax=ax[0])
im = mask_plot(np.angle(modprofile), ax[1])
fig.colorbar(im, ax=ax[1])
ax[0].set_title('$AM(x,y)$')
ax[0].set_ylabel('y, mm')
ax[1].set_title('$PM(x,y)$')
ax[1].set_ylabel('y, mm')
ax[1].set_xlabel('x, mm')
plt.tight_layout()
plt.savefig('mod/mod_slm.png', dpi=200)
plt.show()
