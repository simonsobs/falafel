from __future__ import print_function
from enlib import curvedsky as cs, enmap, lensing as enlensing, powspec, bench
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent
from orphics import io,cosmology,lensing,maps,stats

alex = True

def comp_cltt(alex):

    if alex:
        lmax = 5000
        res = 1.0
        camb_root = "/gpfs01/astro/workarea/msyriac/data/act/theory/cosmo2017_10K_acc3"
        sim_name = "fullskyLensedMapUnaberrated_T_00002.fits"
        #sim_name = "fullskyLensedMap_T_00000.fits"
        ksim_name = "kappaMap_00002.fits"
        sim_location = "/gpfs01/astro/workarea/msyriac/data/sims/dw/"
    else:
        lmax = 3000
        res = 1.5
        camb_root = "/astro/u/msyriac/repos/quicklens/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627"
        sim_name = "lensed_map_seed_1.fits"
        ksim_name = "kappa_map_seed_1.fits"
        sim_location = "/gpfs01/astro/workarea/msyriac/data/depot/falafel/"

    theory = cosmology.loadTheorySpectraFromCAMB(camb_root,get_dimensionless=False)
    Xmap = enmap.read_map(sim_location+sim_name)
    ikappa = enmap.read_map(sim_location+ksim_name)

    if alex:
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
        Xmap.wcs = wcs
        ikappa.wcs = wcs



    calm = cs.map2alm(Xmap,lmax=lmax).astype(np.complex128)
    kalm = cs.map2alm(ikappa,lmax=lmax).astype(np.complex128)
    cls = hp.alm2cl(calm)
    clskk = hp.alm2cl(kalm)

    lsc = np.arange(0,len(cls),1)

    print(cls[:5])
    print(clskk[:5])

    cls[lsc<2] = np.nan
    clskk[lsc<2] = np.nan

    bin_edges = np.logspace(np.log10(2),np.log10(lmax),40)
    binner = stats.bin1D(bin_edges)

    ls = np.arange(0,lmax,1)
    cltt = theory.lCl('TT',ls)
    clkk = theory.gCl('kk',ls)
    cents,btt = binner.binned(ls,cltt)
    cents,bcc = binner.binned(lsc,cls)

    cents,bttkk = binner.binned(ls,clkk)
    cents,bcckk = binner.binned(lsc,clskk)

    # plot
    pl = io.Plotter(yscale='log',xscale='log',xlabel='$\\ell$',ylabel='$C_{\\ell}$')
    pl.add(ls,ls**2.*theory.lCl('TT',ls),lw=3,color='k')
    pl.add(cents,cents**2.*btt,ls="none",marker="x")
    pl.add(cents,cents**2.*bcc,ls="none",marker="o")
    pl.done(io.dout_dir+"fullsky_sim_test_cmb_alex_"+str(alex)+".png")

    pl = io.Plotter(yscale='log',xscale='log',xlabel='$L$',ylabel='$C_L$')
    pl.add(ls,theory.gCl('kk',ls),lw=3,color='k')
    pl.add(cents,bttkk,ls="none",marker="x")
    pl.add(cents,bcckk,ls="none",marker="o")
    pl.done(io.dout_dir+"fullsky_sim_test_kk_alex_"+str(alex)+".png")


    pl = io.Plotter(xscale='log',xlabel='$L$',ylabel='$\\delta C_L / C_L$')
    pl.add(cents,(bcc-btt)/btt,ls="-",marker="o",label="cmb")
    pl.add(cents,(bcckk-bttkk)/bttkk,ls="-",marker="o",label="kk")
    pl.hline()
    pl._ax.set_ylim(-0.2,0.2)
    pl.done(io.dout_dir+"fullsky_sim_test_alex_diff_"+str(alex)+"_2.png")




comp_cltt(alex)
