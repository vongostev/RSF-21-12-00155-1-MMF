# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:56:11 2021

@author: von.gostev
"""

import numpy as np
import matplotlib.pyplot as plt

npoints = 256
bounds = [-12, 12]
loaded_data = np.load('smf_simc_SI_6_properties.npz')
# loaded_data = np.load('mmf_PCF_3.6_properties.npz')
modes = loaded_data['modes_list'][::-1][100:110]

fig, axes = plt.subplots(2, 5, dpi=200, figsize=(10, 4))
for mode, ax in zip(modes, axes.flatten()):
    ax.imshow(np.real(mode).reshape((400, -1)),
              extent=bounds * 2)
plt.tight_layout()
plt.savefig('pcf/pcf_modes.png', dpi=200)
# plt.savefig('simc/smf_simc_modes.png', dpi=200)

plt.show()