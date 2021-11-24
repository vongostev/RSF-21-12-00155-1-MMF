# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 00:42:37 2021

@author: vonGostev
"""

import numpy as np
import matplotlib.pyplot as plt


area_size = 3.5 * 31.25
npoints = 256
bounds = [-area_size / 2, area_size / 2]
max_si_mode = np.load('mmf_SI_50_properties.npz')['modes_list'][-1]
max_grin_mode = np.load('mmf_GRIN_62.5_properties.npz')['modes_list'][-1]
# loaded_data = np.load('mmf_PCF_3.6_properties.npz')

fig, axes = plt.subplots(1, 2, dpi=200, figsize=(8, 4))
axes[0].imshow(np.abs(max_si_mode).reshape((256, -1)),
               extent=bounds * 2)
axes[0].add_patch(plt.Circle((0, 0), 25, color='w', fill=False))
axes[0].set_title('Step-index')
axes[0].set_xlabel('(а)')
axes[1].imshow(np.abs(max_grin_mode).reshape((256, -1)),
               extent=bounds * 2)
axes[1].add_patch(plt.Circle((0, 0), 31.25, color='w', fill=False))
axes[1].set_title('GRIN')
axes[1].set_xlabel('(б)')
plt.tight_layout()
# plt.savefig('pcf/pcf_modes.png', dpi=200)
plt.savefig('mmf/mmf_si_grin_max_modes.png', dpi=200)

plt.show()