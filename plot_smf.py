# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:47:47 2021

@author: vonGostev
"""
import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    return x / np.sum(x)


area_size = 6 * 1.5
bounds = [-area_size / 2, area_size / 2]
xlabel = 'Поперечная координата, мкм'
npoints = 2 ** 8
np_half = 2 ** 7

fiber_data = np.load('coherence_data_171121_smf.npz', allow_pickle=True)
fiber_data = fiber_data['SI'].tolist()

plt.plot(np.linspace(*bounds, 2 ** 8), fiber_data['index'][2**7])
plt.xlabel(xlabel)
plt.ylabel('Показатель преломления')
plt.tight_layout()
plt.savefig('smf/smf_index.png', dpi=200)
plt.show()

fig, ax = plt.subplots(1, 2, dpi=200)
ax[0].imshow(fiber_data['s__ip'], extent=bounds*2)
ax[1].imshow(fiber_data['i__ip'], extent=bounds*2)
ax[0].set_title('Исходный профиль\nинтенсивности')
ax[1].set_title('Профиль интенсивности\nпосле разложения\nпо модам волокна')
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel(xlabel)
ax[1].set_xlabel(xlabel)
plt.tight_layout()
plt.savefig('smf/smf_init_input.png', dpi=200)
plt.show()

np_plot = 128

for k in fiber_data:
    if k.startswith('o__ip'):
        d = float(k.split('_')[-1])
        plt.plot(np.linspace(0, area_size / npoints * np_plot, np_plot),
                  fiber_data[k][np_half, np_half:np_half + np_plot], label=f'{d} см')
plt.legend(frameon=0)
plt.ylabel('Интенсивность, у.е.')
plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
plt.savefig('smf/smf_output.png', dpi=200)
plt.show()

plt.plot(np.linspace(0, area_size / npoints * np_plot, np_plot),
         fiber_data['s__cf'][np_half, np_half:np_half + np_plot], label='исх.')
plt.plot(np.linspace(0, area_size / npoints * np_plot, np_plot),
         fiber_data['i__cf'][np_half, np_half:np_half + np_plot], label='вх.')
for k in fiber_data:
    if k.startswith('o__cf'):
        d = float(k.split('_')[-1])
        plt.plot(np.linspace(0, area_size / npoints * np_plot, np_plot),
                 fiber_data[k][np_half, np_half:np_half + np_plot], label=f'{d} см')
plt.legend(frameon=0)
plt.ylabel('Коэффициент корреляции')
plt.xlabel('Поперечная координата, мкм')
plt.tight_layout()
plt.savefig('smf/smf_output_cf.png', dpi=200)
plt.show()
