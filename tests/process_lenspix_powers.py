from __future__ import print_function
import numpy as np
import glob, os, sys
from orphics import io,cosmology


# load theory
camb_root = "data/cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(camb_root,get_dimensionless=False)


sim_location = "/gpfs01/astro/workarea/msyriac/data/sims/msyriac/lenspix/"


l2000_cpowers = []
l3000_cpowers = []


for i in range(200):
    print(i)
    ls2,cr2 = np.loadtxt(sim_location+"cosmo2017_test_lmax2250_lmax2250_nside2048_interp1.51recon_cross_power_%s.dat" % i,usecols=[0,2],unpack=True)
    l2000_cpowers.append(cr2)

    ls3,cr3 = np.loadtxt(sim_location+"cosmo2017_test_lmax3250_nside2048_interp1.51recon_cross_power_%s.dat" % i,usecols=[0,2],unpack=True)
    l3000_cpowers.append(cr3)

l2c = np.array(l2000_cpowers)
l3c = np.array(l3000_cpowers)[:,:l2c.shape[1]]

ls3 = ls3[:l2c.shape[1]]

c2 = l2c.mean(axis=0)
c3 = l3c.mean(axis=0)

assert np.all(np.isclose(ls2,ls3))

ells = ls2

clkk = theory.gCl('kk',ells) * 4. /( 2. *np.pi) #*((ells+1.)/ells)**2.
pl = io.Plotter(yscale='log',xscale='log',xlabel='',ylabel='')
pl.add(ells,clkk)
pl.add(ells,c2)
pl.add(ells,c3)
pl.done(io.dout_dir+"lenspix_cpower.png")


pl = io.Plotter(xlabel='',ylabel='')
pl.add(ells,(c2-clkk)/clkk)
pl.add(ells,(c3-clkk)/clkk,ls="--")
pl.done(io.dout_dir+"lenspix_cpower_diff.png")
