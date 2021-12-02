# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:49:19 2021

@author: vonGostev
"""

import numpy as np
import matplotlib.pyplot as plt


def V(wl, a, NA):
    return 2 * np.pi / wl * a * NA


def rcorr(a, modesnum):
    return 2 * a * np.log(modesnum) / modesnum


def labeled_scatter(x, y, labels, fiber_type, my=0.85):
    plt.scatter(x, y, label=fiber_type)
    for lb, xi, yi in zip(labels, x, y):
        plt.annotate(lb, (xi + 2, yi * my))


wl = 0.632

grin_a = np.array([25, 31.25])
grin_NA = np.array([0.2, 0.275])
grin_labels = ['GIF50C', 'GIF625']
grin_M = V(wl, grin_a, grin_NA) ** 2 / 4 / 2

si_a = np.array([25, 52.5, 100])
si_NA = 0.22
si_labels = ['FG050', 'FG105', 'FG200']
si_M = V(wl, si_a, si_NA) ** 2 / 2 / 2

fig = plt.figure(figsize=(5, 4))
labeled_scatter(grin_a * 2, grin_M, grin_labels, 'GRIN')
labeled_scatter(si_a * 2, si_M, si_labels, 'Step-Index')
plt.xlabel('Диаметр сердцевины волокна, мкм.')
plt.ylabel('Число мод')
plt.yscale('log')
plt.xlim(30, 230)
plt.ylim(200, 20000)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('mmf/mmf_modesnum.png', dpi=200)
plt.show()

fig = plt.figure(figsize=(5, 4))
labeled_scatter(grin_a * 2, rcorr(grin_a, grin_M), grin_labels, 'GRIN', 1)
labeled_scatter(si_a * 2, rcorr(si_a, si_M), si_labels, 'Step-Index', 0.9)
plt.xlabel('Диаметр сердцевины волокна, мкм.')
plt.ylabel('Радиус корреляции, мкм.')
plt.xlim(30, 230)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('mmf/mmf_rcorr.png', dpi=200)
plt.show()
