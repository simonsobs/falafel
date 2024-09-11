from __future__ import print_function
from orphics import maps,io,cosmology,mpi # msyriac/orphics ; pip install -e . --user
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot,powspec
import numpy as np
import os,sys
import os
import glob
import traceback
import healpy as hp
from . import qe
import math

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

closest_nside = lambda lmax: 1<<( int((lmax/3000.)*2048.)  -1).bit_length()

# These functions are copied from solenspipe, so will have to update
# solenspipe to call them from here

def get_cmb_alm(i,iset,path=config['signal_path']):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullskyLensedUnabberatedCMB_alm_set%s_%s.fits" % (sstr,istr)
    return hp.read_alm(fname,hdu=(1,2,3))


def get_kappa_alm(i,path=config['signal_path']):
    istr = str(i).zfill(5)
    fname = path + "fullskyPhi_alm_%s.fits" % istr
    return plensing.phi_to_kappa(hp.read_alm(fname))


def get_theory_dicts(nells=None,lmax=9000,grad=True):
    thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
    ls = np.arange(lmax+1)
    ucls = {}
    tcls = {}
    theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
    ells,gt,ge,gb,gte = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1,2,3,4])
    if nells is None: nells = {'TT':0,'EE':0,'BB':0}
    ucls['TT'] = maps.interp(ells,gt)(ls) if grad else theory.lCl('TT',ls)
    ucls['TE'] = maps.interp(ells,gte)(ls) if grad else theory.lCl('TE',ls)
    ucls['EE'] = maps.interp(ells,ge)(ls) if grad else theory.lCl('EE',ls)
    ucls['BB'] = maps.interp(ells,gb)(ls) if grad else theory.lCl('BB',ls)
    ucls['kk'] = theory.gCl('kk',ls)
    tcls['TT'] = theory.lCl('TT',ls) + nells['TT']
    tcls['TE'] = theory.lCl('TE',ls)
    tcls['EE'] = theory.lCl('EE',ls) + nells['EE']
    tcls['BB'] = theory.lCl('BB',ls) + nells['BB']
    return ucls, tcls

def get_theory_dicts_white_noise(beam_fwhm,noise_t,noise_p=None,lmax=9000,grad=True):
    ls = np.arange(lmax+1)
    if noise_p is None: noise_p = np.sqrt(2.)*noise_t
    nells = {}
    nells['TT'] = (noise_t*np.pi/180./60.)**2. / maps.gauss_beam(beam_fwhm,ls)**2.
    nells['EE'] = (noise_p*np.pi/180./60.)**2. / maps.gauss_beam(beam_fwhm,ls)**2.
    nells['BB'] = (noise_p*np.pi/180./60.)**2. / maps.gauss_beam(beam_fwhm,ls)**2.
    return get_theory_dicts(nells=nells,lmax=lmax,grad=grad)

def get_theory_dicts_fnoise(fnoise_t,fnoise_e=None,fnoise_b=None,lmax=9000,grad=True,scale_noise=1.):
    ls = np.arange(lmax+1)
    nells = {}
    nells['TT'] = scale_noise*fnoise_t(ls)
    nells['EE'] = scale_noise*fnoise_e(ls) if not(fnoise_e is None) else scale_noise*nells['TT']*2.
    nells['BB'] = scale_noise*fnoise_b(ls) if not(fnoise_b is None) else scale_noise*nells['EE']
    return get_theory_dicts(nells=nells,lmax=lmax,grad=grad)


def change_alm_lmax(alms, lmax, dtype=np.complex128):
    ilmax  = hp.Alm.getlmax(alms.shape[-1])
    olmax  = lmax
    oshape     = list(alms.shape)
    oshape[-1] = hp.Alm.getsize(olmax)
    oshape     = tuple(oshape)
    alms_out   = np.zeros(oshape, dtype = dtype)
    flmax      = min(ilmax, olmax)
    for m in range(flmax+1):
        lminc = m
        lmaxc = flmax
        idx_isidx = hp.Alm.getidx(ilmax, lminc, m)
        idx_ieidx = hp.Alm.getidx(ilmax, lmaxc, m)
        idx_osidx = hp.Alm.getidx(olmax, lminc, m)
        idx_oeidx = hp.Alm.getidx(olmax, lmaxc, m)
        alms_out[..., idx_osidx:idx_oeidx+1] = alms[..., idx_isidx:idx_ieidx+1].copy()
    return alms_out


# performs isotropic (or Wiener) filtering with / without TE mode mixing
def isotropic_filter(alm,ucls,tcls,lmin,lmax,ignore_te=True,wiener=False):
    ucltt, tcltt = ucls['TT'], tcls['TT']
    uclte, tclte = ucls['TE'], tcls['TE']
    uclee, tclee = ucls['EE'], tcls['EE']
    uclbb, tclbb = ucls['BB'], tcls['BB']

    if ignore_te:
        filt_T, filt_E, filt_B = tcltt*0, tclee*0, tclbb*0
        with np.errstate(divide='ignore', invalid='ignore'):
            filt_T[2:] = 1./tcltt[2:]
            filt_E[2:] = 1./tclee[2:]
            filt_B[2:] = 1./tclbb[2:]
            if wiener:
                filt_T[2:] *= ucltt[2:]
                filt_E[2:] *= uclee[2:]
                filt_B[2:] *= uclbb[2:]
        talm = qe.filter_alms(alm[0],filt_T,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[1],filt_E,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_B,lmin=lmin,lmax=lmax)

    else:
        filt_TT, filt_TE, filt_ET, filt_EE = tcltt*0, tclte*0, tclte*0, tclee*0
        filt_BB = tclbb*0

        with np.errstate(divide='ignore', invalid='ignore'):
            # det of TT + EE block (aka prefactor of its inverse)
            te_det = 1. / (tcltt[2:]*tclee[2:] - tclte[2:]**2.)

            filt_TT[2:] = te_det
            filt_EE[2:] = te_det
            filt_TE[2:] = te_det
            filt_ET[2:] = te_det
            filt_BB[2:] = 1. / tclbb[2:]

            if wiener:
                filt_TT[2:] *= (ucltt[2:]*tclee[2:] - uclte[2:]*tclte[2:])
                filt_EE[2:] *= (uclee[2:]*tcltt[2:] - uclte[2:]*tclte[2:])
                filt_BB[2:] *= uclbb[2:]
                # these two are no longer symmetric 
                filt_TE[2:] *= (uclte[2:]*tcltt[2:] - ucltt[2:]*tclte[2:])
                filt_ET[2:] *= (uclte[2:]*tclee[2:] - uclee[2:]*tclte[2:])
            else:
                filt_TT[2:] *= tclee[2:]
                filt_EE[2:] *= tcltt[2:]
                filt_TE[2:] *= -tclte[2:]
                filt_ET[2:] *= -tclte[2:]

        talm = qe.filter_alms(alm[0],filt_TT,lmin=lmin,lmax=lmax) + \
               qe.filter_alms(alm[1],filt_TE,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[0],filt_ET,lmin=lmin,lmax=lmax) + \
               qe.filter_alms(alm[1],filt_EE,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_BB,lmin=lmin,lmax=lmax)
        
    return [talm,ealm,balm]