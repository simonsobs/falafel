from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
from orphics import maps,io,cosmology,mpi # msyriac/orphics ; pip install -e . --user
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
import numpy as np
import os,sys
import os
import glob
import traceback
import healpy as hp
from . import qe

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

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


def isotropic_filter(alm,tcls,lmin,lmax,ignore_te=True):
    # Filter isotropically
    tcltt = tcls['TT']
    tclee = tcls['EE']
    tclte = tcls['TE']
    tclbb = tcls['BB']
    if ignore_te:
        filt_T = tcltt*0
        filt_E = tclee*0
        filt_B = tclbb*0
        filt_T[2:] = 1./tcltt[2:]
        filt_E[2:] = 1./tclee[2:]
        filt_B[2:] = 1./tclbb[2:]
        talm = qe.filter_alms(alm[0],filt_T,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[1],filt_E,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_B,lmin=lmin,lmax=lmax)
    else:
        filt_T_T = tcltt*0
        filt_E_T = tclee*0
        filt_T_E = tcltt*0
        filt_E_E = tclee*0
        filt_B = tclbb*0
        filt_T_T[2:] = tclee[2:]/(tcltt[2:]*tclee[2:] - tclte[2:]**2.)
        filt_T_E[2:] = -tclte[2:]/(tcltt[2:]*tclee[2:] - tclte[2:]**2.)
        filt_E_T[2:] = -tclte[2:]/(tcltt[2:]*tclee[2:] - tclte[2:]**2.)
        filt_E_E[2:] = tcltt[2:]/(tcltt[2:]*tclee[2:] - tclte[2:]**2.)
        filt_B[2:] = 1./tclbb[2:]
        talm = qe.filter_alms(alm[0],filt_T_T,lmin=lmin,lmax=lmax) + qe.filter_alms(alm[1],filt_T_E,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[0],filt_E_T,lmin=lmin,lmax=lmax) + qe.filter_alms(alm[1],filt_E_E,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_B,lmin=lmin,lmax=lmax)
        
    return [talm,ealm,balm]
