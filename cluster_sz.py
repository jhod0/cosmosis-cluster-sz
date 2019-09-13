#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
from cluster_toolkit import pressure as pp
from cosmosis.datablock import BlockError
import numpy as np
from scipy.interpolate import interp1d

blkname = 'cluster_sz'


# TODO:
#   * properly handle M200m v M200c
#   * finish 2halo


def setup(options):
    config = {}
    # Choose a type of 1halo profile
    config['type'] = options[blkname, 'profiletype']

    if config['type'] == 'battaglia':
        # Battaglia pressure params
        A_P_0 = options.get_double(blkname, 'A_P_0', default=18.1)
        am_P_0 = options.get_double(blkname, 'am_P_0', default=0.154)
        az_P_0 = options.get_double(blkname, 'az_P_0', default=-0.758)
        config['params_P_0'] = (A_P_0, am_P_0, az_P_0)

        # Battaglia scale radius params
        A_x_c = options.get_double(blkname, 'A_x_c', default=0.497)
        am_x_c = options.get_double(blkname, 'am_x_c', default=-0.00865)
        az_x_c = options.get_double(blkname, 'az_x_c', default=0.731)
        config['params_x_c'] = (A_x_c, am_x_c, az_x_c)

        # Battaglia beta shape params
        A_beta = options.get_double(blkname, 'A_beta', default=4.35)
        am_beta = options.get_double(blkname, 'am_beta', default=0.0393)
        az_beta = options.get_double(blkname, 'az_beta', default=0.415)
        config['params_beta'] = (A_beta, am_beta, az_beta)

        battaglia_opts = ['batt_alpha', 'batt_gamma']
        # Best-fits from BBPS2
        battaglia_defaults = [1, -0.3]

        for o, dflt in zip(battaglia_opts, battaglia_defaults):
            try:
                config[o] = options.get_double(blkname, o, default=dflt)
            except BlockError:
                config[o] = options.get_int(blkname, o, default=dflt)
    else:
        msg = 'Invalid cluster pressure profile ({type}), must be one of {opts}'
        raise ValueError(msg.format(type=config['type'],
                                    opts=['battaglia']))

    # Observational parameters
    config['thetas'] = options[blkname, 'thetas']
    config['sigma_psf'] = options[blkname, 'sigma_psf']

    Mmin = options[blkname, 'Mmin']
    Mmax = options[blkname, 'Mmax']
    nM = options.get_int(blkname, 'nM', default=20)
    config['Ms'] = np.geomspace(Mmin, Mmax, nM)

    zmin = options[blkname, 'zmin']
    zmax = options[blkname, 'zmax']
    nz = options.get_int(blkname, 'nz', default=20)
    config['zs'] = np.linspace(zmin, zmax, nz)

    return config


def execute(sample, config):
    # Take in cosmological parameters
    dist_zs = sample['distances', 'z']
    das_interp = interp1d(dist_zs, sample['distances', 'd_a'])

    omb = sample['cosmological_parameters', 'omega_b']
    omm = sample['cosmological_parameters', 'omega_m']
    h0 = sample['cosmological_parameters', 'h0']

    # Sort out profile parameters
    if config['type'] == 'battaglia':
        profile_cls = pp.BBPSProfile
        profile_params = {'omega_b': omb,
                          'omega_m': omm,
                          'h': h0,
                          'params_P_0': config['params_P_0'],
                          'params_x_c': config['params_x_c'],
                          'params_beta': config['params_beta'],
                          'alpha': config['batt_alpha'],
                          'gamma': config['batt_gamma']}
    else:
        msg = 'cluster_sz internal error: unknown profile {}'
        raise ValueError(msg.format(config['type']))

    # Find grid to compute 1halos, 2halos at
    sigma_psf = config['sigma_psf']
    thetas = config['thetas']
    Ms = config['Ms']
    zs = config['zs']

    # Compute 1 and 2halo for each (theta, m, z)
    ys_1h = np.zeros((thetas.size, Ms.size, zs.size))
    ys_2h = np.zeros_like(ys_1h)

    for iz, z in enumerate(zs):
        da = das_interp(z)
        # Compute 2halo
        # (TODO)
        th = pp.TwoHaloProfile(omb, omm, h0,
                               # hmb_m, hmb_z, hmb_b,
                               # hmf_m, hmf_z, hmf_f,
                               # P_lin_k, P_lin_z, P_lin,
                               # mdelta_m, mdelta_c,
                               one_halo=profile_cls,
                               one_halo_kwargs=profile_params)
        # TODO figure out good k spacing
        ks = np.geomspace(0.1, 10, 20)
        mass_indep_2h = th.convolved_y_FT(thetas, ks, z, da,
                                          sigma_beam=sigma_psf)

        for iM, M in enumerate(Ms):
            oh = profile_cls(M, z, **profile_params)

            # Compute 1halo
            theta_max = np.sqrt(2) * thetas.max()
            thetas_interp = interp1d(*oh.convolved_y(da, theta_max,
                                                     sigma=sigma_psf))
            ys_1h[:, iM, iz] = thetas_interp(thetas)
            # TODO compute bias_at_M
            ys_2h[:, iM, iz] = bias_at_M * mass_indep_2h

    sample[blkname, 'ys_2h'] = ys_2h
    sample[blkname, 'ys_1h'] = ys_1h

    sample[blkname, 'zs'] = zs
    sample[blkname, 'Ms'] = Ms
    sample[blkname, 'thetas'] = thetas

    return 0


def cleanup(config):
    # nothing to do here
    return 0
