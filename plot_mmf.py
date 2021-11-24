# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:10:51 2021

@author: von.gostev
"""

import numpy as np
import matplotlib.pyplot as plt


def _n(x):
    return x / np.max(x)

def _c(y, X, r=25):
    y[np.abs(X) > r] = 0
    return y



xlabel = 'Поперечная координата, мкм'
npoints = 2 ** 8
np_half = 2 ** 7
index_type = 'GRIN'
mod_type = 'dmdslm'
fiber_type = 'mmf'
data_dir = 'mmf'
_loaded_data = np.load(
    f'{data_dir}/cohdata_241121_{fiber_type}_{mod_type}.npz', allow_pickle=True)
loaded_data = _loaded_data[index_type].tolist()

area_size = loaded_data['params']['area_size']
bounds = [-area_size / 2, area_size / 2]
core_radius = loaded_data['params']['core_radius']

# plt.plot(np.linspace(*bounds, 2 ** 8), loaded_data['index'][2**7])
# plt.xlabel(xlabel)
# plt.ylabel('Показатель преломления')
# plt.tight_layout()
# plt.savefig(f'{fiber_type}/{fiber_type}_{index_type.lower()}_index.png', dpi=200)
# plt.show()

# np_plot = 128
X = np.linspace(*bounds, npoints)
# plt.plot(X, _n(loaded_data['s__ip'][np_half]), label='исх.')
# plt.plot(X, _n(loaded_data['i__ip'][np_half]), label='вх.')
# for k in loaded_data:
#     if k.startswith('o__ip'):
#         d = float(k.split('_')[-1])
#         plt.plot(X, _n(loaded_data[k][np_half]), label=f'{d} см')
# plt.legend(frameon=0)
# plt.ylabel('Интенсивность, у.е.')
# plt.xlabel('Поперечная координата, мкм')
# plt.tight_layout()
# plt.savefig(f'{fiber_type}/{fiber_type}_{index_type.lower()}_{mod_type}_output.png', dpi=200)
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
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
        ax[i].imshow(loaded_data[k], label=f'вых. {d} см', extent=bounds*2)
        ax[i].set_xlabel(f'({lbl[i - 2]})')
        i += 1
    if i == 3:
        break
        
plt.legend(frameon=0)
# plt.ylabel('Интенсивность, у.е.')
# plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
plt.savefig(f'{fiber_type}/{fiber_type}_{index_type.lower()}_{mod_type}_ip.png', dpi=200)
plt.show()

# X = np.linspace(0, area_size / npoints * np_plot, np_plot)
plt.plot(X, _c(_n(loaded_data['s__cf'][np_half]), X, r=core_radius), label='исх.')
plt.plot(X, _c(_n(loaded_data['i__cf'][np_half]), X, r=core_radius), label='вх.')
for k in loaded_data:
    # if k.startswith('o__cf_0'):
    if k == 'o__cf_0':
        d = float(k.split('_')[-1])
        plt.plot(X, _c(_n(loaded_data[k][np_half]), X, r=core_radius), 
                 label='вых.' + (f' {d} см' if d > 0 else ''))
plt.legend(frameon=0)
plt.ylabel('Коэффициент корреляции')
plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
plt.savefig(f'{fiber_type}/{fiber_type}_{index_type.lower()}_{mod_type}_output_cf.png', dpi=200)
plt.show()

# fig, ax = plt.subplots(2, 3, figsize=(9, 6))
# ax = ax.flatten()
# ax[0].imshow(loaded_data['s__cf'], label='исх.', extent=bounds*2)
# ax[0].set_xlabel('(a)')
# ax[1].imshow(loaded_data['i__cf'], label='вх.', extent=bounds*2)
# ax[1].set_xlabel('(b)')

# i = 2
# lbl = 'cdef'
# for k in loaded_data:
#     if k.startswith('o__cf'):
#         d = float(k.split('_')[-1])
#         ax[i].imshow(loaded_data[k], label=f'{d} см', extent=bounds*2)
#         ax[i].set_xlabel(f'({lbl[i - 2]})')
#         i += 1
        
# plt.legend(frameon=0)
# # plt.ylabel('Интенсивность, у.е.')
# # plt.xlabel('Поперечная координата, мкм')
# plt.tight_layout()
# plt.savefig(f'{fiber_type}/{fiber_type}_{index_type.lower()}_{mod_type}_cf_2d.png', dpi=200)
# plt.show()