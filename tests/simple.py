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
ests = ['TT','mv','mvpol','EE','TE','EB','TB']
#ests = ['mv']
#ests = ['TT']

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
xucls,tcls = utils.get_theory_dicts_white_noise(beam_fwhm,noise_t)


# Get normalizations
Als = pytempura.get_norms(ests,ucls,tcls,lmin,lmax,k_ellmax=mlmax)

# Filter isotropically
tcltt = tcls['TT']
tclee = tcls['EE']
tclbb = tcls['BB']

filt_T = tcltt*0
filt_E = tclee*0
filt_B = tclbb*0

filt_T[2:] = 1./tcltt[2:]
filt_E[2:] = 1./tclee[2:]
filt_B[2:] = 1./tclbb[2:]

talm = qe.filter_alms(alm[0],filt_T,lmin=lmin,lmax=lmax)
ealm = qe.filter_alms(alm[1],filt_E,lmin=lmin,lmax=lmax)
balm = qe.filter_alms(alm[2],filt_B,lmin=lmin,lmax=lmax)


# Reconstruct
recon = qe.qe_all(px,ucls,mlmax,
                  fTalm=talm,fEalm=ealm,fBalm=balm,
                  estimators=ests,
                  xfTalm=talm,xfEalm=ealm,xfBalm=balm)
    
# Get input kappa alms
ikalm = utils.change_alm_lmax(utils.get_kappa_alm(sindex).astype(np.complex128),mlmax)


# Cross-correlate and plot
kalms = {}
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))
bin_edges = np.geomspace(2,mlmax,15)
print(bin_edges)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)
print(ells.shape)
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
