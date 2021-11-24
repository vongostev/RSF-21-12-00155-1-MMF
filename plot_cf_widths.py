# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:55:34 2021

@author: vonGostev
"""
import numpy as np
import matplotlib.pyplot as plt
from lightprop2d import Beam2D, um, mm


def _n(x):
    return x / np.max(x)


def _c(y, X, r=-1):
    if r > 0:
        y[np.abs(X) > r] = 0
    return y


SAVE = False
xlabel = 'Поперечная координата, мкм'
npoints = 2 ** 8
np_half = 2 ** 7
index_type = 'GRIN'
mod_type = 'slm'
fiber_type = 'mmf'
data_dir = 'mmf'
_loaded_data = np.load(
    f'{data_dir}/cohdata_241121_{fiber_type}_{mod_type}.npz', allow_pickle=True)
loaded_data = _loaded_data[index_type].tolist()

area_size = loaded_data['params']['area_size']
bounds = [-area_size / 2, area_size / 2]
core_radius = loaded_data['params']['core_radius']
wl = 0.632

distances = np.array([0, 100 * um, 1000 * um, 1])
expands = [1, 2, 2, 4]

cfs = [v for k, v in loaded_data.items() if k.startswith('o__cf')]

widths = []
for i, cf in enumerate(cfs):
    _L = area_size * np.prod(expands[:i+1])
    width = np.sum(cf[np_half] >= 1 / np.exp(1)) / npoints * _L
    widths.append(width)

exp_distances = np.array([20, 40, 60, 80, 100, 150, 200, 3000]) * um
exp_rx = np.array([11.2, 11.4, 12.0, 12.7, 13.8, 16.3, 20.5, 84.15]) * 5.2 / 25
th_distances = np.sort(np.concatenate((distances, exp_distances)))
plt.plot(th_distances, widths[0] + wl * th_distances / um / 31.25 * 0.16, '--', color='black', label='Theory')
plt.scatter(distances, widths, color='r', label='Calc.')
plt.scatter(exp_distances, exp_rx, label='Exp.')
plt.xscale('log')
plt.xlabel('Расстояние, см.')
plt.ylabel('Радиус корреляции, мкм')
plt.legend(frameon=0)
plt.tight_layout()
plt.savefig('mmf/mmf_grin_cf_distance_exp.png', dpi=200)
plt.show()
