# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:10:51 2021

@author: von.gostev
"""

import numpy as np
import matplotlib.pyplot as plt


def _n(x):
    return x / np.max(x)


def _c(y, X, r=-1):
    if r > 0:
        y[np.abs(X) > r] = 0
    return y


SAVE = 1
xlabel = 'Поперечная координата, мкм'
npoints = 2 ** 8
np_half = 2 ** 7
index_type = 'SI'
mod_type = 'dmd'
fiber_type = 'mmf'
data_dir = '50_62.5'
_loaded_data = np.load(
    f'{data_dir}/cohdata_191121_{fiber_type}_{mod_type}.npz', allow_pickle=True)
loaded_data = _loaded_data[index_type].tolist()

area_size = loaded_data['params']['area_size']
bounds = [-area_size / 2, area_size / 2]
core_radius = loaded_data['params']['core_radius']

plt.plot(np.linspace(*bounds, 2 ** 8), loaded_data['index'][2**7])
plt.xlabel(xlabel)
plt.ylabel('Показатель преломления')
plt.tight_layout()
if SAVE:
    plt.savefig(
        f'{data_dir}/{fiber_type}_{index_type.lower()}_index.png', dpi=100)
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax = ax.flatten()
ax[0].imshow(loaded_data['s__ip'], label='исх.', extent=bounds*2)
ax[0].set_xlabel('(a)')
ax[1].imshow(loaded_data['i__ip'], label='вх.', extent=bounds*2)
ax[1].set_xlabel('(b)')
ax[2].imshow(loaded_data['o__ip_0'], label='вых.', extent=bounds*2)
ax[2].set_xlabel('(c)')
plt.legend(frameon=0)
plt.tight_layout()
if SAVE:
    plt.savefig(
        f'{data_dir}/{fiber_type}_{index_type.lower()}_{mod_type}_ip.png', dpi=150)
plt.show()

X = np.linspace(*bounds, npoints)
plt.plot(X, _c(_n(loaded_data['s__cf'][np_half]),
               X, r=core_radius), label='исх.')
plt.plot(X, _c(_n(loaded_data['i__cf'][np_half]),
               X, r=core_radius), label='вх.')
plt.plot(X, _c(_n(loaded_data['o__cf_0'][np_half]), X, r=core_radius),
         label='вых.')
plt.legend(frameon=0)
plt.ylabel('Коэффициент корреляции')
plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
if SAVE:
    plt.savefig(
        f'{data_dir}/{fiber_type}_{index_type.lower()}_{mod_type}_output_cf.png', dpi=100)
plt.show()
