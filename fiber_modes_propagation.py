# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 19:43:56 2021

@author: vonGostev
"""

import numpy as np
import matplotlib.pyplot as plt
from lightprop2d import Beam2D, gaussian_beam, um

npoints = 256
area_size = 3.5 * 31.25
wl = 0.632
loaded_data = np.load('mmf_GRIN_62.5_properties.npz')[
    'modes_list'].tolist()
# loaded_data = np.load('mmf_PCF_3.6_properties.npz')
# for mode in loaded_data:
#     mode = np.array(mode)
#     beam = Beam2D(area_size=area_size, wl=wl, npoints=npoints,
#                   init_field=mode.reshape((npoints, -1)))
#     plt.plot(beam.iprofile[npoints // 2])
#     beam.propagate(100)
#     plt.plot(beam.iprofile[npoints // 2])
#     plt.show()

beam = Beam2D(area_size=area_size, wl=wl, npoints=npoints,
              init_field=np.array(loaded_data[0]).reshape((npoints, -1)))
dz = 0.002
z_grid = np.arange(0, 200, 10) * dz
widths = []

# beam.lens(2)
for z in z_grid:
    if z > z_grid[0]:
        beam.propagate(dz / um)
    widths.append(beam.D4sigma[0] / 2)


gbeam = Beam2D(area_size=area_size, wl=wl, npoints=npoints,
               init_field_gen=gaussian_beam, init_gen_args=(1, widths[0] / 1.36))
gwidths = []

# beam.lens(2)
for z in z_grid:
    if z > z_grid[0]:
        gbeam.propagate(dz / um)
    gwidths.append(gbeam.D4sigma[0] / 2)

plt.plot(z_grid, widths)
plt.plot(z_grid, gwidths)
plt.plot(z_grid, wl * z_grid / um / widths[0])
plt.show()
