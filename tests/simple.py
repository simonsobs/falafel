from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe,utils
import pytempura

# The estimators to test lensing for
#ests = ['TT','mv','mvpol','EE','TE','EB','TB']
ests = ['mv']
# ests = ['TT']

# Decide on a geometry for the intermediate operations
res = 2.0 # resolution in arcminutes
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
px = qe.pixelization(shape,wcs)

# Choose sim index
sindex = 1

# Maximum multipole for alms
mlmax = 4000

# Filtering configuration
lmax = 3000
lmin = 100
beam_fwhm = 0.
noise_t = 0.

# Get CMB alms
alm = utils.get_cmb_alm(sindex,0)

# Get theory spectra
ucls,tcls = utils.get_theory_dicts_white_noise(beam_fwhm,noise_t)


# Get normalizations
Als = pytempura.get_norms(ests,ucls,ucls,tcls,lmin,lmax,k_ellmax=mlmax)

# Filter
Xdat = utils.isotropic_filter(alm,tcls,lmin,lmax)

# Reconstruct
recon = qe.qe_all(px,ucls,mlmax,
                  fTalm=Xdat[0],fEalm=Xdat[1],fBalm=Xdat[2],
                  estimators=ests,
                  xfTalm=Xdat[0],xfEalm=Xdat[1],xfBalm=Xdat[2])
    
# Get input kappa alms
ikalm = utils.change_alm_lmax(utils.get_kappa_alm(sindex).astype(np.complex128),mlmax)


# Cross-correlate and plot
kalms = {}
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))
bin_edges = np.geomspace(2,mlmax,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)
for est in ests:
    pl = io.Plotter('CL')
    pl2 = io.Plotter('rCL',xyscale='loglin')
    kalms[est] = plensing.phi_to_kappa(hp.almxfl(recon[est][0].astype(np.complex128),Als[est][0] )) # ignore curl
    pl.add(ells,(ells*(ells+1.)/2.)**2. * Als[est][0],ls='--')
    cls = hp.alm2cl(kalms[est],ikalm)
    acls = hp.alm2cl(kalms[est],kalms[est])
    pl.add(ells,acls,label='r x r')
    pl.add(ells,cls,label = 'r x i')
    pl.add(ells,icls, label = 'i x i')
    pl2.add(*bin((cls-icls)/icls),marker='o')
    pl2.hline(y=0)
    pl2._ax.set_ylim(-0.1,0.1)
    pl2.done(f'simple_recon_diff_{est}.png')
    pl._ax.set_ylim(1e-9,1e-5)
    pl.done(f'simple_recon_{est}.png')
