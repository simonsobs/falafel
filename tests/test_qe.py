from __future__ import print_function
from enlib import curvedsky as cs, enmap, lensing as enlensing, powspec, bench
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent
from orphics import io,cosmology,lensing,maps,stats
from falafel import qe

use_saved = True
res = 1.0 # resolution in arcminutes
cache = True

lmax = 2000 # cmb ellmax
mlmax = 2*lmax # lmax used for harmonic transforms
dtype = np.float64


seed = 1

# res = 1.0 # resolution in arcminutes
# lmax = 3000 # cmb ellmax
# mlmax = 2*lmax # lmax used for harmonic transforms
# dtype = np.float32


# load theory
if use_saved:
    camb_root = "/gpfs01/astro/workarea/msyriac/data/act/theory/cosmo2017_10K_acc3"
else:
    camb_root = "/astro/u/msyriac/repos/quicklens/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627"
theory = cosmology.loadTheorySpectraFromCAMB(camb_root,get_dimensionless=False)

# ells corresponding to modes in the alms
ells = np.arange(0,mlmax,1)
ntt = nee = nbb = ells*0. # no noise

ellmin_t = 2
ellmax_t = lmax

#### NORM FROM FLAT-SKY CODE FOR NOW
bin_edges = np.linspace(2,lmax,40)
with bench.show("flat sky AL"):
    ls,nlkks,theory,qest = lensing.lensing_noise(ells,ntt,nee,nbb, \
                                                 ellmin_t,ellmin_t,ellmin_t, \
                                                 ellmax_t,ellmax_t,ellmax_t, \
                                                 bin_edges, \
                                                 camb_theory_file_root=None, \
                                                 estimators = ['TT'], \
                                                 delens = False, \
                                                 theory=theory, \
                                                 dimensionless=False, \
                                                 unlensed_equals_lensed=True, \
                                                 grad_cut=None,width_deg=25.,px_res_arcmin=res)
    
binner = stats.bin2D(qest.N.modLMap,bin_edges)
cents,albinned = binner.bin(qest.AL['TT'])
Al = maps.interp(cents,albinned)(ells)
Nl = maps.interp(ls,nlkks['TT'])(ells)
lpls,lpal = np.loadtxt("nls.txt",unpack=True)
pl = io.Plotter(yscale='log',xscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
pl.add(ells,Nl,ls="--")
pl.add(lpls,lpal*(lpls*(lpls+1.))**2./4.,ls="-.")
#pl._ax.set_ylim(1e-10,1e-6)
pl.done(io.dout_dir+"fullsky_qe_result_al.png")

dh_nls = np.nan_to_num(lpal*(lpls*(lpls+1.))**2./4.)
dh_als = np.nan_to_num(dh_nls * 2. / lpls /(lpls+1))
Al = dh_als

#### MAKE FULL-SKY LENSED SIMS

shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
if not(use_saved):
    # Make a full-sky enlib geometry in CAR
    pfile = camb_root+"_lenspotentialCls.dat"
    ps = powspec.read_camb_full_lens(pfile).astype(dtype)
    cache_loc = "/gpfs01/astro/workarea/msyriac/data/depot/falafel/"
    try:
        assert cache
        Xmap = enmap.read_map("%slensed_map_seed_%d.fits" % (cache_loc,seed))
        ikappa = enmap.read_map("%skappa_map_seed_%d.fits" % (cache_loc,seed))
        print("Loaded cached maps...")
    except:
        print("Making random lensed map...")
        with bench.show("lensing"):
            Xmap,ikappa, = enlensing.rand_map(shape[-2:], wcs, ps, lmax=mlmax,dtype=dtype,output="lk",seed=seed)
        if cache:
            enmap.write_map("%slensed_map_seed_%d.fits" % (cache_loc,seed),Xmap)
            enmap.write_map("%skappa_map_seed_%d.fits" % (cache_loc,seed),ikappa)
            print("Maps have been cached.")

else:
    print("Loading lensed map...")
    sim_location = "/gpfs01/astro/workarea/msyriac/data/sims/dw/"
    Xmap = enmap.read_map(sim_location+"fullskyLensedMapUnaberrated_T_00001.fits")
    ikappa = enmap.read_map(sim_location+"kappaMap_00001.fits")
    Xmap.wcs = wcs
    ikappa.wcs = wcs
    assert Xmap.shape==shape
    assert ikappa.shape==shape


### DO FULL SKY RECONSTRUCTION
print("Calculating unnormalized full-sky kappa...")
lcltt = theory.lCl('TT',range(lmax))
with bench.show("reconstruction"):
    ukappa_alm = qe.qe_tt_simple(Xmap,lcltt=lcltt,nltt_deconvolved=0.,lmin=2,lmax=lmax)

# alms of input kappa
ik_alm = cs.map2alm(ikappa,lmax=mlmax).astype(np.complex128)
# if use_saved:
#     p2k = ells*(ells+1.)/2.
#     ik_alm = hp.almxfl(ik_alm,p2k)
# alms of reconstruction
kappa_alm = hp.almxfl(ukappa_alm,Al).astype(np.complex128)

# cross and auto powers
cri = hp.alm2cl(ik_alm,kappa_alm)
crr = hp.alm2cl(kappa_alm)
cii = hp.alm2cl(ik_alm)

ls = np.arange(len(crr))

cri[ls<2] = np.nan
crr[ls<2] = np.nan
cii[ls<2] = np.nan


bin_edges = np.logspace(np.log10(2),np.log10(mlmax),40)
binner = stats.bin1D(bin_edges)
cents,bri = binner.binned(ls,cri)
cents,brr = binner.binned(ls,crr)
cents,bii = binner.binned(ls,cii)

# plot
pl = io.Plotter(yscale='log',xscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
# pl.add(ells,Nl,ls="--")
pl.add(cents,bri,ls="none",marker="o")
pl.add(cents,brr,ls="none",marker="o")
pl.add(cents,bii,ls="none",marker="x")
pl.add(lpls,lpal*(lpls*(lpls+1.))**2./4.,ls="-.")
pl.done(io.dout_dir+"fullsky_qe_result.png")

pl = io.Plotter(xscale='log')
pl.add(cents,(bri-bii)/bii,ls="-",marker="o")
pl.hline()
pl._ax.set_ylim(-0.1,0.1)
pl.done(io.dout_dir+"fullsky_qe_result_diff.png")
