# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:01:03 2021

@author: vonGostev
"""

import numpy as np
import matplotlib.pyplot as plt


def _n(x):
    return x / np.max(x)


area_size = 3.5 * 25
bounds = [-area_size / 2, area_size / 2]
xlabel = 'Поперечная координата, мкм'
npoints = 2 ** 8
np_half = 2 ** 7
fiber_type = 'GRIN'
mod_type = 'amplphase'

_loaded_data = np.load('coherence_data_191121_mmf_amplphase.npz', allow_pickle=True)
loaded_data = _loaded_data[fiber_type].tolist()

plt.plot(np.linspace(*bounds, 2 ** 8), loaded_data['index'][2**7])
plt.xlabel(xlabel)
plt.ylabel('Показатель преломления')
plt.tight_layout()
plt.savefig(f'mmf/mmf_{fiber_type.lower()}_index.png', dpi=200)
plt.show()

fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].imshow(loaded_data['s__ip'], extent=bounds*2)
ax[1].imshow(loaded_data['i__ip'], extent=bounds*2)
ax[0].set_title('Исходный профиль\nинтенсивности')
ax[1].set_title('Профиль интенсивности\nпосле разложения\nпо модам волокна')
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel(xlabel)
ax[1].set_xlabel(xlabel)
plt.tight_layout()
plt.savefig(f'mmf/mmf_{fiber_type.lower()}_{mod_type}_init_input.png', dpi=200)
plt.show()

np_plot = 128
X = np.linspace(*bounds, npoints)
plt.plot(X, _n(loaded_data['s__ip'][np_half]), label='исх.')
plt.plot(X, _n(loaded_data['i__ip'][np_half]), label='вх.')
for k in loaded_data:
    if k.startswith('o__ip'):
        d = float(k.split('_')[-1])
        plt.plot(X, _n(loaded_data[k][np_half]), label=f'{d} см')
plt.legend(frameon=0)
plt.ylabel('Интенсивность, у.е.')
plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
plt.savefig(f'mmf/mmf_{fiber_type.lower()}_{mod_type}_output.png', dpi=200)
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
ax = ax.flatten()
ax[0].imshow(loaded_data['s__ip'], label='исх.', extent=bounds*2)
ax[0].set_xlabel('(a)')
ax[1].imshow(loaded_data['i__ip'], label='вх.', extent=bounds*2)
ax[1].set_xlabel('(b)')

i = 2
lbl = 'cdef'
for k in loaded_data:
    if k.startswith('o__ip'):
        d = float(k.split('_')[-1])
        ax[i].imshow(loaded_data[k], label=f'{d} см', extent=bounds*2)
        ax[i].set_xlabel(f'({lbl[i - 2]})')
        i += 1
        
plt.legend(frameon=0)
# plt.ylabel('Интенсивность, у.е.')
# plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
plt.savefig(f'mmf/mmf_{fiber_type.lower()}_{mod_type}_ip.png', dpi=200)
plt.show()

X = np.linspace(0, area_size / npoints * np_plot, np_plot)
plt.plot(X, _n(loaded_data['s__cf']
         [np_half, np_half:np_half + np_plot]), label='исх.')
plt.plot(X, _n(loaded_data['i__cf']
         [np_half, np_half:np_half + np_plot]), label='вх.')
for k in loaded_data:
    if k.startswith('o__cf'):
        d = float(k.split('_')[-1])
        plt.plot(X, _n(loaded_data[k][np_half,
                 np_half:np_half + np_plot]), label=f'{d} см')
plt.legend(frameon=0)
plt.ylabel('Коэффициент корреляции')
plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
plt.savefig(f'mmf/mmf_{fiber_type.lower()}_{mod_type}_output_cf.png', dpi=200)
plt.show()
