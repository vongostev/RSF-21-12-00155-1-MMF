# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:48:17 2021

@author: vonGostev
"""
import __init__
import numpy as np
import matplotlib.pyplot as plt
from lightprop2d import Beam2D, um, plane_wave, round_hole, gaussian_beam, random_round_hole_phase
from scipy.linalg import expm


npoints = 256
cradius = 31.25
mod_radius = cradius / 2
area_size = cradius * 3.5
wl = 0.632
distances = [0, 100 * um, 1000 * um, 1]
bounds = [- area_size / 2, area_size / 2]
# loaded_data = np.load('smf_SI_3_properties.npz')
# loaded_data = np.load('mmf_SI_50_properties.npz')
loaded_data = np.load('mmf_GRIN_62.5_properties.npz')
modes = loaded_data['modes_list']
OP = loaded_data['fiber_op']
ibeam = Beam2D(area_size=area_size * um, npoints=npoints, wl=wl * um, unsafe_fft=1,
               init_field_gen=gaussian_beam,
               init_gen_args=(1, mod_radius * um,),
               use_gpu=0)
# init_field_gen=round_hole, init_gen_args=(cradius,))
profiles = [ibeam.iprofile]

mc = ibeam.deconstruct_by_modes(modes)
ibeam.construct_by_modes(modes, mc)
# ibeam.expand(ibeam.area_size * 2)

profiles.append(ibeam.iprofile)

mc = expm(1j * OP * 10.) @ mc

for d in distances:
    ibeam.construct_by_modes(modes, mc)
    ibeam.propagate(d)
    profiles.append(ibeam.iprofile)
    ibeam.expand(ibeam.area_size * 2)
    ibeam.coarse(2)

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
axes = ax.flatten()

i = 0
lbl = 'abcdef'
for ax, p in zip(axes, profiles):
    if i < 3:
        ax.imshow(p, extent=bounds*2)
    else:
        ax.imshow(p, extent=[- area_size / 2 * (i - 1),
                             area_size / 2 * (i - 1)]*2)
    ax.set_xlabel(f'({lbl[i]})')
    i += 1
plt.tight_layout()
plt.savefig('mmf/mmf_grin_gaussian_ip.png', dpi=100)
plt.show()

loaded_data.close()
