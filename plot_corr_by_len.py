# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:37:02 2021

@author: vonGostev
"""
import __init__
import numpy as np
import matplotlib.pyplot as plt


area_size = 3.5 * 31.25
bounds = [-area_size / 2, area_size / 2]
xlabel = 'Поперечная координата, мкм'
npoints = 2 ** 8
np_half = 2 ** 7
wl = 0.632
NA = 0.275
a = 31.25
index_type = 'SI'
mod_type = 'slm'
fiber_type = 'mmf'
_loaded_data = np.load(
    f'mmf/corrlen_data_211121_mmf_{mod_type}.npz', allow_pickle=True)
loaded_data = _loaded_data[index_type].tolist()['fl__cf']


widths = []
for ip in loaded_data:
    widths.append(np.sum(ip[np_half] > 0.5) * area_size / npoints)
widths = np.array(widths)
Z = np.linspace(0, 23, 15)

plt.plot(Z, widths)
# plt.plot(Z, widths[:, 1])

plt.plot(Z, 2 * wl / a / NA / np.pi * np.sin(NA * Z))
