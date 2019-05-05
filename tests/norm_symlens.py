from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
import symlens

thloc = "/scratch/r/rbond/msyriac/data/sims/alex/v0.4/cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

def get_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=100,lmax=2000,plot=True):
    shape,wcs = maps.rect_geometry(width_deg=80.,px_res_arcmin=2.0*3000./lmax)
    emin = maps.minimum_ell(shape,wcs)
    modlmap = enmap.modlmap(shape,wcs)
    tctt = maps.interp(range(len(tctt)),tctt)(modlmap)
    uctt = maps.interp(range(len(uctt)),uctt)(modlmap)
    tcee = maps.interp(range(len(tcee)),tcee)(modlmap)
    ucee = maps.interp(range(len(ucee)),ucee)(modlmap)
    tcbb = maps.interp(range(len(tcbb)),tcbb)(modlmap)
    ucbb = maps.interp(range(len(ucbb)),ucbb)(modlmap)
    ucte = maps.interp(range(len(ucte)),ucte)(modlmap)
    tcte = maps.interp(range(len(tcte)),tcte)(modlmap)
    
    tmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    feed_dict = {
        'uC_T_T':uctt,
        'tC_T_T':tctt,
        'uC_E_E':ucee,
        'tC_E_E':tcee,
        'uC_T_E':ucte,
        'tC_T_E':tcte,
        'tC_B_B':tcbb,
        'uC_B_B':ucbb,
    }
    polcombs = ['TT','TE','EE','TB','EB']
    Als = {}
    Al1ds = {}
    bin_edges = np.arange(3*emin,lmax,2*emin)
    binner = stats.bin2D(modlmap,bin_edges)
    alinv_mv = 0.
    alinv_mv_pol = 0.
    for pol in polcombs:
        Al = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator="hu_ok", XY=pol, xmask=tmask, ymask=tmask)
        cents,Al1d = binner.bin(Al)
        ls = np.arange(0,cents.max(),1)
        Als[pol] = np.interp(ls,cents,Al1d*cents**2.)/ls**2.
        Als[pol][ls<1] = 0
        Al1ds[pol] = Al1d.copy()
        alinv_mv += (1./Als[pol])
        if pol=='EE' or pol=='EB': alinv_mv_pol += (1./Als[pol])
    al_mv = (1./alinv_mv)
    al_mv[ls<1] = 0
    al_mv_pol = (1./alinv_mv_pol)
    al_mv_pol[ls<1] = 0
    if plot:
        pl = io.Plotter(xyscale='loglog',xlabel='',ylabel='')
        pl.add(ells,clkk,color='k',lw=3)
        pl.add(ls,al_mv*ls**2.,ls="-",color="red",label='mv',lw=2)
        pl.add(ls,al_mv_pol*ls**2.,ls="-",color="green",label='mv_pol',lw=2)
        for i,pol in enumerate(polcombs):
            pl.add(cents,Al1ds[pol]*cents**2.,color="C%d" % i,label=pol)
            pl.add(ls,Als[pol]*ls**2.,ls="--",color="C%d" % i)
        pl.done()


    Al1 = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator="hdv", XY="TE", xmask=tmask, ymask=tmask)
    Al2 = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator="hdv", XY="ET", xmask=tmask, ymask=tmask)
    Al_te_hdv = 1./((1./Al1)+(1./Al2))
    cents,Al1d = binner.bin(Al_te_hdv)
    Al_te_hdv = np.interp(ls,cents,Al1d*cents**2.)/ls**2.
    Al_te_hdv[ls<1] = 0
        
    return ls,Als,al_mv_pol,al_mv,Al_te_hdv


ells = np.arange(3100)
uctt = tctt = theory.lCl('TT',ells)
ucee = tcee = theory.lCl('EE',ells)
ucbb = tcbb = theory.lCl('BB',ells)
ucte = tcte = theory.lCl('TE',ells)
clkk = theory.gCl('kk',ells)

ls,Als,al_mv_pol,al_mv,Al_te_hdv = get_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=100,lmax=3000,plot=False)
io.save_cols("norm.txt",(ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv))
