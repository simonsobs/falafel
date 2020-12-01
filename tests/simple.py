from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,curvedsky as cs,lensing
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe

"""
build d(T)
build d(E,B)
build d(E=0,B)
build d(E,B=0)

TT = G(d(T))
EE = G(d(E,B=0))
EB = G(d(E,B)) - EE
TB = G(d(T) + d(E=0,B)) - TT
TE = G(d(T) + d(E,B=0)) - EE - TT 
MV = G(d(T) + d(E,B))
MV_pol = G(d(E,B))

TT
EE
EE + EB
TT + EE + TE + EB + TB
"""


est = "temp"
#est = "pol"
#est = "mv"

thloc = "/home/msyriac/data/act/theory/cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

res = 2.0 # resolution in arcminutes
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
#shape,wcs = enmap.band_geometry((np.deg2rad(-60.),np.deg2rad(30.)),res=np.deg2rad(res/60.), proj="car")
sim_location = "/home/msyriac/data/sims/alex/v0.4/"
sindex = str(1).zfill(5)
mlmax = 4000

alm = maps.change_alm_lmax(
    hp.read_alm("/home/msyriac/data/sims/alex/v0.4/fullskyLensedUnabberatedCMB_alm_set00_%s.fits"
                % sindex,hdu=(1,2,3))
    ,mlmax)

lmax = 3000
lmin = 100

talm = alm[0]
ealm = alm[1]
balm = alm[2]


falm = qe.filter_alms(talm,lambda x: 1./theory.lCl('TT',x),lmin,lmax)
xalm = qe.filter_alms(talm,lambda x: 1,lmin,lmax)
X_Ealm = qe.filter_alms(ealm,lambda x: 1,lmin,lmax)
X_Balm = qe.filter_alms(balm,lambda x: 1,lmin,lmax)
Y_Ealm = qe.filter_alms(ealm,lambda x: 1./theory.lCl('EE',x),lmin,lmax)
Y_Balm = qe.filter_alms(balm,lambda x: 1./theory.lCl('BB',x),lmin,lmax)

px = qe.pixelization(shape,wcs)

if est=="temp":
    with bench.show("recon tt"): recon = qe.qe_temperature_only(px,xalm,falm,mlmax)[0]
elif est=="pol":
    with bench.show("recon pol"): recon = qe.qe_pol_only(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax)[0]
    with bench.show("recon pol"): recon_ee = qe.qe_pol_only(px,X_Ealm,X_Balm*0,Y_Ealm,Y_Balm*0,mlmax)[0]
elif est=="mv":
    with bench.show("recon mv"): recon = qe.qe_mv(px,xalm,X_Ealm,X_Balm,falm,Y_Ealm,Y_Balm,mlmax)[0]

palm = maps.change_alm_lmax(hp.read_alm(sim_location+"fullskyPhi_alm_%s.fits" % (sindex)),mlmax)
ikalm = lensing.phi_to_kappa(palm)

ls,Als,Als_ee,Als_eb,Als_te,Als_tb,al_mv_pol,al_mv,Al_te_hdv = np.loadtxt("norm.txt",unpack=True)
if est=="pol":
    Als = al_mv_pol # !!
    kalms_eb = hp.almxfl(recon-recon_ee,Als_eb)
elif est=="mv":
    Als = al_mv
# Als = Als_ee # !!
kalms = hp.almxfl(recon,Als)

cls = hp.alm2cl(kalms,ikalm)
icls = hp.alm2cl(ikalm,ikalm)
acls = hp.alm2cl(kalms,kalms)
if est=='pol':
    cls_eb = hp.alm2cl(kalms_eb,ikalm)
    acls_eb = hp.alm2cl(kalms_eb,kalms_eb)
    

pl = io.Plotter(xyscale='loglog')#lin',scalefn=lambda x: x)
ells = np.arange(len(icls))
if est=='pol':
    pl.add(ells,acls_eb,label='EB')
    pl.add(ells,cls_eb,label='EB',marker='o')
pl.add(ells,acls)
pl.add(ls,maps.interp(ells,icls)(ls) + (Als*ls*(ls+1)/4.))
pl.add(ells,cls)
pl.add(ells,icls)
pl.add(ells,theory.gCl('kk',ells))
pl.hline(y=0)
pl._ax.set_ylim(1e-9,1e-5)
pl.done()
