# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:33:40 2021

@author: von.gostev
"""
import __init__

from logging import Logger, StreamHandler, Formatter
from joblib import Parallel, delayed
from pyMMF import (IndexProfile, propagationModeSolver,
                   estimateNumModesGRIN, estimateNumModesSI)
from lightprop2d import (
    Beam2D, random_round_hole_phase, random_round_hole, um,
    plane_wave, random_wave_bin, random_round_hole_bin)
from dataclasses import dataclass, field
from scipy.linalg import expm
import sys
from time import perf_counter
import numpy as np
import cupy as cp


log = Logger('coh.random')

handler = StreamHandler(sys.stdout)
handler.setLevel(10)
formatter = Formatter(
    '%(asctime)s - %(name)-10.10s [%(levelname)-7.7s]  %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

"""
1. Распространение случайного поля в волокне и за ним при амплитудной модуляции
    - Картинка маски DMD
    - Картинка профиля до волокна 
    - Картинка профиля на входе в волокно (собранного из мод)
    - Картинка модового состава
    - Картинка поля на выходе из волокна
    - Корреляционная функция исходного поля и корреляционная функция поля на
    входе в волокно
    - Корреляционная функция при фиксированной длине волокна
    - Зависимость ширины корреляционной функции от длины волокна

2. Распространение случайного поля в волокне и за ним при фазовой модуляции
    - Картинка маски SLM
    - Картинка профиля до волокна 
    - Картинка профиля на входе в волокно (собранного из мод)
    - Картинка модового состава
    - Картинка поля на выходе из волокна
    - Корреляционная функция исходного поля и корреляционная функция поля на
    входе в волокно
    - Корреляционная функция при фиксированной длине волокна
    - Зависимость ширины корреляционной функции от длины волокна

3. Распространение случайного поля в волокне и за ним при амплитудно-фазовой модуляции
    - Картинка амплитудно-фазовой маски
    - Картинка профиля до волокна 
    - Картинка профиля на входе в волокно (собранного из мод)
    - Картинка модового состава
    - Картинка поля на выходе из волокна
    - Корреляционная функция исходного поля и корреляционная функция поля на
    входе в волокно
    - Корреляционная функция при фиксированной длине волокна
    - Зависимость ширины корреляционной функции от длины волокна

"""


def fiber_data_gen(obj, fiber_len, prop_distance):
    obj.init_beam()
    _iprofile = obj.get_output_iprofile(fiber_len)
    if prop_distance > 0:
        obj.propagate(prop_distance)
        return obj.prop_iprofile
    else:
        return _iprofile


@dataclass
class LightFiberAnalyser:

    area_size: float = 40
    npoints: int = 2**8
    wl: float = 0.632
    init_field_gen: object = plane_wave
    init_gen_args: tuple = ()

    NA: float = 0.2
    n1: float = 1.45
    core_radius: float = 15

    simc__kwargs: dict = \
        field(
            default_factory=lambda: dict(
                core_pitch=5,
                dims=1,
                layers=1,
                delta=0.039,
                central_core=True)
        )
    pcf__kwargs: dict = \
        field(
            default_factory=lambda: dict(
                core_pitch=5,
                dims=1,
                layers=1,
                delta=0.039,
                cladding_radius=25)
        )

    index_type: str = 'GRIN'
    fiber_type: str = 'mmf'
    curvature: object = None
    solver_mode: str = 'eig'
    fiber_len: float = 0

    use_gpu: bool = True
    nimg: int = 1000

    mod_radius: float = 0

    def __post_init__(self):

        profile = self.__set_index_profile()
        self.solver = propagationModeSolver()
        self.solver.setIndexProfile(profile)
        self.solver.setWL(self.wl)

        self.fiber_props = \
            f"{self.fiber_type}_{self.index_type}_{self.core_radius * 2:g}_properties.npz"

    def __set_index_profile(self):
        profile = IndexProfile(npoints=self.npoints,
                               areaSize=self.area_size)
        if self.index_type == 'GRIN':
            profile.initParabolicGRIN(
                n1=self.n1, a=self.core_radius, NA=self.NA)
            self.Nmodes_estim = estimateNumModesGRIN(
                self.wl, self.core_radius, self.NA, pola=1)
        elif self.index_type == 'SI':
            profile.initStepIndex(
                n1=self.n1, a=self.core_radius, NA=self.NA)
            self.Nmodes_estim = estimateNumModesSI(
                self.wl, self.core_radius, self.NA, pola=1)
        elif self.index_type == 'SIMC':
            profile.initStepIndexMultiCoreRadial(
                n1=self.n1, a=self.core_radius, NA=self.NA, **self.simc__kwargs)
            self.Nmodes_estim = estimateNumModesSI(
                self.wl, self.core_radius, self.NA, pola=1) * (
                    self.simc__kwargs['dims'] * self.simc__kwargs['layers'])
        elif self.index_type == 'PCF':
            self.Nmodes_estim = 1000
            profile.initPhotonicCrystalHex(
                n1=self.n1, a=self.core_radius, NA=self.NA, **self.pcf__kwargs)
        return profile

    def set_modulation_func(self, mod_func, *args, **kwargs):
        self._mf = lambda x, y: mod_func(x, y, *args, **kwargs)

    def fiber_calc(self):
        t = perf_counter()
        try:
            with np.load(self.fiber_props) as data:
                self.fiber_op = data["fiber_op"]
                self.modes = cp.array(data["modes_list"])
                self.betas = data["betas"]
            log.info(f'Fiber data loaded from `{self.fiber_props}`')
        except FileNotFoundError:
            if self.index_type == 'SI':
                modes = self.solver.solve(mode='SI', n_jobs=-2)
            else:
                modes = self.solver.solve(
                    nmodesMax=self.Nmodes_estim+10, boundary='close',
                    mode='eig', curvature=None, propag_only=True)
            self.modes = cp.array(modes.profiles)[
                np.argsort(modes.betas)[::-1]]
            self.betas = np.array(np.sort(modes.betas))[::-1]
            self.fiber_op = modes.getEvolutionOperator()
            np.savez_compressed(self.fiber_props,
                                fiber_op=self.fiber_op,
                                modes_list=self.modes,
                                betas=self.betas)
            log.info(f'Fiber data saved to `{self.fiber_props}`')

        modes_matrix = cp.array(np.real(np.vstack(self.modes).T))
        self.modes_matrix_t = cp.array(modes_matrix.T)
        self.modes_matrix_dot_t = modes_matrix.T.dot(modes_matrix)
        log.info(f'Found {len(self.modes)} modes')
        log.info(f"Fiber initialized. Elapsed time {perf_counter() - t:.3f} s")
        self.modes_coeffs = np.zeros((self.nimg, len(self.modes)),
                                     dtype=np.complex128)

    def init_beam(self):
        self.beam = Beam2D(self.area_size, self.npoints, self.wl,
                           init_field_gen=self.init_field_gen,
                           init_gen_args=self.init_gen_args,
                           unsafe_fft=1, use_gpu=self.use_gpu)
        self.mask = self._mf(self.beam.X, self.beam.Y)
        self.beam.coordinate_filter(f_init=self.mask)

    def propagate(self, z=0):
        self.beam.propagate(z)

    def get_index_profile(self):
        return self.solver.indexProfile.n.reshape([self.npoints] * 2)

    def set_transmission_matrix(self, fiber_len):
        self.tm = cp.array(expm(1j * self.fiber_op * fiber_len))

    @property
    def iprofile(self):
        return self.beam.iprofile

    def get_input_iprofile(self):
        return self.get_output_iprofile(0)

    def get_output_iprofile(self, fiber_len=0, mc=None):
        if mc is None:
            mc = self.beam.fast_deconstruct_by_modes(
                self.modes_matrix_t, self.modes_matrix_dot_t)
        if fiber_len > 0:
            mc = self.tm @ self.beam._asxp(mc)
        self.beam.construct_by_modes(self.modes, mc)
        return self.iprofile

    def _get_cf(self, obj_data, ref_data, parallel_njobs=-1, fast=False):
        t = perf_counter()
        if fast:
            log.info(
                'Compute correlation function fast using `np.tensordot`')
            self.cf = np.tensordot(obj_data - obj_data.mean(),
                                   ref_data - ref_data.mean(axis=0),
                                   axes=1) / np.sqrt(obj_data.std() ** 2 * ref_data.std() ** 2)
        else:
            log.info(
                'Compute correlation function slow but exact using `np.corrcoef`')

            def gi(pixel_data):
                return np.nan_to_num(np.corrcoef(obj_data, pixel_data))[0, 1]

            img_shape = ref_data.shape[1:]
            ref_data = ref_data.reshape(ref_data.shape[0], -1).T
            corr_data = Parallel(n_jobs=parallel_njobs)(
                delayed(gi)(s) for s in ref_data)
            self.cf = np.asarray(corr_data).reshape(img_shape)
        log.info(
            f"Correlation function calculated. Elapsed time {perf_counter() - t:.3f} s")

    def correlate_init(self, nimg=0):
        t = perf_counter()
        idata = np.zeros((nimg, self.npoints, self.npoints))
        for i in range(nimg):
            self.init_beam()
            idata[i, :, :] = self.iprofile
        point_data = idata[:, self.npoints // 2, self.npoints // 2]
        log.info(
            f"Initial data to cf generated. Elapsed time {perf_counter() - t:.3f} s")
        self._get_cf(point_data, idata)

        return self.cf

    def correlate_input(self, nimg=0):
        nimg = self.nimg if nimg == 0 else nimg
        t = perf_counter()
        idata = np.zeros((nimg, self.npoints, self.npoints))
        for i in range(nimg):
            self.init_beam()
            _iprofile = self.get_output_iprofile(0)
            idata[i, :, :] = _iprofile
            self.modes_coeffs[i, :] = self.beam._np(self.beam.modes_coeffs)
        point_data = idata[:, self.npoints // 2, self.npoints // 2]
        log.info(
            f"In-fiber data to cf generated. Elapsed time {perf_counter() - t:.3f} s")
        self._get_cf(point_data, idata)
        return self.cf

    def correlate_output(self, nimg=0, fiber_len=0, prop_distance=0):
        nimg = self.nimg if nimg == 0 else nimg
        t = perf_counter()
        idata = np.zeros((nimg, self.npoints, self.npoints))
        for i in range(nimg):
            _iprofile = self.get_output_iprofile(
                fiber_len, self.modes_coeffs[i])
            if prop_distance > 0:
                self.propagate(prop_distance)
                idata[i, :, :] = self.iprofile
            else:
                idata[i, :, :] = _iprofile

        point_data = idata[:, self.npoints // 2, self.npoints // 2]
        log.info(
            f"In-fiber data to cf generated. Elapsed time {perf_counter() - t:.3f} s")
        self._get_cf(point_data, idata)
        return self.cf

    def correlate_by_fiber_len(self, nimg=0, max_fiber_len=1 / um):
        for l in np.linspace(0, max_fiber_len, 10):
            self.correlate_output(nimg, l, 0)


# um
fiber_len = 10 / um  # um for cm
distances = [0, 100 * um, 1000 * um, 1]
n_cf = 1000

fiber_params = [
    # dict(
    #     area_size=3.5 * 31.25,  # um
    #     # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=358
    #     index_type='GRIN',
    #     core_radius=31.25,
    #     NA=0.275,
    #     # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
    #     n1=1.4613,
    #     mod_radius=31.25
    # ),
    # dict(
    #     area_size=3.5 * 31.25,  # um
    #     # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=6838
    #     index_type='SI',
    #     core_radius=25,
    #     NA=0.22,
    #     # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
    #     n1=1.4613,
    #     mod_radius=25
    # ),
    dict(
        area_size = 4 * 3,  # um
        # to SIMC IXF-MC-12-PAS-6
        index_type='SI',
        fiber_type='smf_simc',
        core_radius=3,
        NA=0.19,
        # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
        n1=1.4613,
        mod_radius=3
    ),
    # dict(
    #     area_size = 6 * 1.5,  # um
    #     # https://www.thorlabs.com/drawings/b2e64c24c4214c42-DB6047F0-B8E1-ED20-62AA1F597ADBE2AB/S405-XP-SpecSheet.pdf
    #     index_type='SI',
    #     fiber_type='smf',
    #     core_radius=1.5,
    #     NA=0.12,
    #     # https://www.frontiersin.org/articles/10.3389/fnins.2019.00082/full#B13
    #     n1=1.4613,
    #     mod_radius=1.5
    # ),
    # dict(
    #     area_size=3.5 * 31.25,
    #     npoints=256,
    #     # https://photonics.ixblue.com/sites/default/files/2021-06/IXF-MC-12-PAS-6_edA_multicore_fiber.pdf
    #     index_type='SIMC',
    #     core_radius=3,
    #     NA=0.19,
    #     n1=1.4613,
    #     simc__kwargs=dict(
    #         core_pitch=35,
    #         delta=0.039,
    #         dims=12,
    #         layers=1,
    #         central_core_radius=0
    #     ),
    #     mod_radius=42
    # ),
    # dict(
    #     # https://www.thorlabs.com/thorproduct.cfm?partnumber=S405-XP
    #     index_type='PCF',
    #     core_radius=1.15,
    #     NA=0.2,
    #     n1=1.45704,
    #     pcf__kwargs=dict(
    #         central_core_radius=3.5,
    #         core_pitch=0.0,
    #         pcf_radius=24,
    #         cladding_radius=65
    #     )
    # )
]

fiber_data = {}

mod_params = {
    'dmd': {
        'init_gen': plane_wave,
        'init_args': (),
        'mod_gen': random_round_hole_bin
    },
    'ampl': {
        'init_gen': plane_wave,
        'init_args': (),
        'mod_gen': random_round_hole
    },
    'slm': {
        'init_gen': plane_wave,
        'init_args': (),
        'mod_gen': random_round_hole_phase
    },
    'dmdslm': {
        'init_gen': random_wave_bin,
        'init_args': (),
        'mod_gen': random_round_hole_phase
    },
}

date = '191121'
data_dir = 'simc'
using_gpu = True
if not using_gpu:
    cp = np
else:
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

for mod in mod_params:
    for params in fiber_params:
        itype = params['index_type']
        log.info(f"Analysing {itype} fiber")
        fiber_data[itype] = {'params': params}
        mod_radius = params['mod_radius']

        analyser = LightFiberAnalyser(
            use_gpu=using_gpu,
            init_field_gen=mod_params[mod]['init_gen'],
            init_gen_args=mod_params[mod]['init_args'],
            **params)
        analyser.set_modulation_func(mod_params[mod]['mod_gen'],
                                     mod_radius, binning_order=1)
        analyser.init_beam()
        # Пример исходного профиля
        fiber_data[itype]['s__ip'] = analyser.iprofile
        # Исходная корреляционная функция
        fiber_data[itype]['s__cf'] = analyser.correlate_init(n_cf)

        analyser.fiber_calc()
        # Профиль показателя преломления
        fiber_data[itype]['index'] = analyser.get_index_profile()
        # Пример профиля на входе волокна после разложения по модам
        fiber_data[itype]['i__ip'] = analyser.get_input_iprofile()
        # Корреляционная функция на входе волокна
        fiber_data[itype]['i__cf'] = analyser.correlate_input(n_cf)

        analyser.set_transmission_matrix(fiber_len)
        log.info(f'Set fiber length to {fiber_len * um} cm')
        for d in distances:
            log.info(f"Set propagation distance to {d:g} cm")
            # Корреляционная функция после волокна на расстоянии d см
            fiber_data[itype][f'o__cf_{d}'] = analyser.correlate_output(
                n_cf, fiber_len, d)
            # Пример профиля после волокна на расстоянии d см
            fiber_data[itype][f'o__ip_{d}'] = analyser.iprofile

    fname = f'{data_dir}/cohdata_{date}_{analyser.fiber_type}_{mod}.npz'
    np.savez_compressed(fname, **fiber_data)
    log.info(f'Data saved to `{fname}`')

    if using_gpu:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
