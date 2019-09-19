#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
from cluster_toolkit import pressure as pp
from colossus.cosmology import cosmology as col_cosmo
from colossus.halo import concentration, mass_defs
from cosmosis.datablock import BlockError
import numpy as np
from scipy.interpolate import interp1d, interp2d


def convert_mass(m, z, mdef_in='200c', mdef_out='200m',
                 concentration_model='diemer19', profile='nfw'):
    '''
    Converts between mass definitions.
    '''
    c = concentration.concentration(m, mdef_in, z,
                                    model=concentration_model,
                                    conversion_profile=profile)
    return mass_defs.changeMassDefinition(m, c, z, mdef_in, mdef_out,
                                          profile=profile)[0]


def package_toolkit_cosmology(sample):
    cosmo = {}

    for name in ['omega_b', 'omega_m', 'h0', 'n_s', 'sigma_8']:
        cosmo[name] = float(sample['cosmological_parameters', name])
    omega_m = cosmo['omega_m']
    h0 = cosmo['h0']

    # TODO make this robust to flat, non-flat, \Lambda- or w-CDM
    col_cosmo.setCosmology('cluster_sz_cosmo',
                           {'Om0': cosmo['omega_m'],
                            'Ob0': cosmo['omega_b'],
                            'H0': 100*cosmo['h0'],
                            'sigma8': cosmo['sigma_8'],
                            'ns': cosmo['n_s']})

    # Load in table of comoving dist vs. redshift
    cosmo['z_chi'] = sample['distances', 'z']
    cosmo['chi'] = sample['distances', 'd_m']
    cosmo['d_a'] = sample['distances', 'd_a']
    cosmo['d_a_i'] = interp1d(cosmo['z_chi'], cosmo['d_a'])

    # Get halo mass function
    # NB: mass definition is _MEAN MASS OVERDENSITY_, not _CRITICAL MASS
    # OVERDENSITY_
    cosmo['hmf_z'] = sample['mass_function', 'z']
    cosmo['hmf_m'] = sample['mass_function', 'm_h'] * omega_m / h0
    cosmo['hmf_f'] = sample['mass_function', 'dndlnmh'] * h0**3

    # (Convert to dn/dm from dn/d(lnm))
    for i in range(cosmo['hmf_f'].shape[0]):
        cosmo['hmf_f'][i, :] /= cosmo['hmf_m']
    cosmo['hmf'] = interp2d(cosmo['hmf_m'], cosmo['hmf_z'], cosmo['hmf_f'])

    # Get the halo mass bias
    # As with HMF, NB: mass definition is _MEAN MASS OVERDENSITY_, not
    # _CRITICAL MASS OVERDENSITY_
    cosmo['hmb_z'] = sample['tinker_bias_function', 'z']
    cosmo['hmb_m'] = np.exp(sample['tinker_bias_function', 'ln_mass_h'])/h0
    cosmo['hmb_b'] = sample['tinker_bias_function', 'bias']
    cosmo['hmb'] = interp2d(cosmo['hmb_m'], cosmo['hmb_z'], cosmo['hmb_b'])

    # Get the matter power spectrum
    cosmo['P_lin_k'] = sample['matter_power_lin', 'k_h'] * h0
    cosmo['P_lin_z'] = sample['matter_power_lin', 'z']
    cosmo['P_lin'] = sample['matter_power_lin', 'p_k'] / (h0**3)
    cosmo['P_lin_i'] = interp2d(cosmo['P_lin_k'], cosmo['P_lin_z'],
                                cosmo['P_lin'])

    return cosmo


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
    # Sort cosmological parameters
    cosmo = package_toolkit_cosmology(sample)
    dist_zs = sample['distances', 'z']
    das_interp = interp1d(dist_zs, sample['distances', 'd_a'])

    # Sort out profile parameters
    if config['type'] == 'battaglia':
        profile_cls = pp.BBPSProfile
        profile_params = {'params_P_0': config['params_P_0'],
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

        # Compute 2halo - only z dependent
        # First, get a mean-to-critical mass definition table which we can use
        # to interpolate the conversion
        m200c_lo = convert_mass(cosmo['hmf_m'].min(), z,
                                mdef_in='200m', mdef_out='200c')
        m200c_hi = convert_mass(cosmo['hmf_m'].max(), z,
                                mdef_in='200m', mdef_out='200c')
        m200c = np.geomspace(m200c_lo * 0.99, m200c_hi * 1.01,
                             cosmo['hmf_m'].size + 2)
        m200m = convert_mass(m200c, z, mdef_in='200c', mdef_out='200m')
        mean_to_crit = interp1d(m200m, m200c)

        th = pp.TwoHaloProfile(cosmo, m200m, m200c,
                               one_halo=profile_cls,
                               one_halo_kwargs=profile_params)

        # Compute K spacing
        nk = 100
        theta_to_r = (da / 60) * (np.pi / 180)
        rmax = thetas.max() * theta_to_r
        rmin = thetas.min() * theta_to_r
        ks = np.geomspace(1 / (4*np.pi*rmax), 2*np.pi/(rmin), nk)
        print('computing 2h at z = {:.3e}...'.format(z))
        # We need a higher epsabs precision for 2halo
        mass_indep_2h = th.convolved_y_FT(thetas, da, z, ks,
                                          sigma_beam=sigma_psf,
                                          epsabs_re=1e-18,
                                          epsrel=1e-1)

        # Iterate over M200m
        for iM, Mm in enumerate(Ms):
            print('computing 1h of M200m = {:.3e}...'.format(Mm))
            # Compute 1halo
            Mc = mean_to_crit(Mm)
            oh = profile_cls(Mc, z, cosmo, **profile_params)
            # Convolve with analytic version - uses proj.-slice thm
            # eps bounds avoid GSL error
            thetamax = 60
            small_thetas = thetas < thetamax
            ys_1h[small_thetas, iM, iz] = oh.convolved_y(thetas[small_thetas],
                                                         da,
                                                         sigma_beam=sigma_psf,
                                                         epsabs=1e-11,
                                                         epsrel=1e-2)
            # TODO compute bias_at_M
            bias_at_M = 1
            ys_2h[:, iM, iz] = bias_at_M * mass_indep_2h

    # Save outputs
    sample[blkname, 'ys_2h'] = ys_2h
    sample[blkname, 'ys_1h'] = ys_1h

    sample[blkname, 'zs'] = zs
    sample[blkname, 'Ms'] = Ms
    sample[blkname, 'thetas'] = thetas

    return 0


def cleanup(config):
    # nothing to do here
    return 0
