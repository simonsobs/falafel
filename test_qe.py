from enlib import curvedsky as cs, enmap, lensing as enlensing, powspec, bench
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent
from orphics import io,cosmology,lensing,maps,stats
import qe

res = 1.0 # resolution in arcminutes
lmax = 3000 # cmb ellmax
mlmax = 2*lmax # lmax used for harmonic transforms
dtype = np.float32

# Make a full-sky enlib geometry in CAR
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")

# load theory
theory = cosmology.loadTheorySpectraFromCAMB("data/Aug6_highAcc_CDM",get_dimensionless=False)

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
                                                 grad_cut=None, \
                                                 shape=shape,wcs=wcs)

binner = stats.bin2D(qest.N.modLMap,bin_edges)
cents,albinned = binner.bin(qest.AL['TT'])
Al = maps.interp(cents,albinned)(ells)
Nl = maps.interp(ls,nlkks['TT'])(ells)



#### MAKE FULL-SKY LENSED SIMS
pfile = "data/Aug6_highAcc_CDM_lenspotentialCls.dat"
ps = powspec.read_camb_full_lens(pfile).astype(dtype)
print("Making random lensed map...")
with bench.show("lensing"):
    Xmap,ikappa, = enlensing.rand_map(shape[-2:], wcs, ps, lmax=mlmax,dtype=dtype,output="lk")
# io.plot_img(Xmap)

### DO FLAT SKY RECONSTRUCTION TO COMPARE (my version of orphics.lensing has an enlib.bench inside that times it)
print("Calculating flat-sky kappa...")
fkappa = qest.kappa_from_map("TT",Xmap)

### DO FULL SKY RECONSTRUCTION
print("Calculating unnormalized full-sky kappa...")
lcltt = theory.lCl('TT',range(lmax))
with bench.show("reconstruction"):
    ukappa_alm = qe.qe_tt_simple(Xmap,lcltt=lcltt,nltt_deconvolved=0.,lmin=2,lmax=lmax)

# alms of input kappa
ik_alm = cs.map2alm(ikappa,lmax=mlmax).astype(np.complex128)
# alms of reconstruction
kappa_alm = hp.almxfl(ukappa_alm,Al).astype(np.complex128)

# cross and auto powers
cri = hp.alm2cl(ik_alm,kappa_alm)
crr = hp.alm2cl(kappa_alm)

# plot
pl = io.Plotter(yscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
pl.add(ells,Nl,ls="--")
pl.add(ells,cri[:-1])
pl.add(ells,crr[:-1])
#pl._ax.set_ylim(1e-10,1e-6)
pl.done(io.dout_dir+"fullsky_qe_result.png")
