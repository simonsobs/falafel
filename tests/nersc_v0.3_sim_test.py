from __future__ import print_function
from enlib import curvedsky as cs, enmap, lensing as enlensing, powspec, bench
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent
from orphics import io,cosmology,lensing,maps,stats,mpi
from falafel import qe
import os,sys


lmax = int(sys.argv[1]) # cmb ellmax
Nsims = int(sys.argv[2])


comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]


res = 1.0 # resolution in arcminutes
mlmax = 2*lmax # lmax used for harmonic transforms



# load theory
camb_root = "data/cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(camb_root,get_dimensionless=False)

# ells corresponding to modes in the alms
ells = np.arange(0,mlmax,1)
    
lpls,lpal = np.loadtxt("data/nls_%d.txt" % lmax,unpack=True)
if rank==0:
    pl = io.Plotter(yscale='log',xscale='log')
    pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
    pl.add(lpls,lpal*(lpls*(lpls+1.))**2./4.,ls="-.")
    pl.done(io.dout_dir+"fullsky_qe_result_al.png")

dh_nls = np.nan_to_num(lpal*(lpls*(lpls+1.))**2./4.)
dh_als = np.nan_to_num(dh_nls * 2. / lpls /(lpls+1))
Al = dh_als

shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
sim_location = "/global/cscratch1/sd/engelen/simsS1516_v0.3/data/"
ksim_location = "/global/cscratch1/sd/dwhan89/shared/act/simsS1516_v0.3/data/"

#bin_edges = np.logspace(np.log10(2),np.log10(mlmax),40)
bin_edges = np.linspace(2,mlmax,200)
binner = stats.bin1D(bin_edges)

mstats = stats.Stats(comm)

for task in my_tasks:


    sim_index = task+1
    assert sim_index>0

    sindex = str(sim_index).zfill(5)
    if rank==0: print("Loading lensed map...")
    Xmap = enmap.read_map(sim_location+"cmb_set00_%s/fullskyLensedMapUnaberrated_T_%s.fits" % (sindex,sindex))
    Xmap.wcs = wcs
    assert Xmap.shape==shape


    ### DO FULL SKY RECONSTRUCTION
    if rank==0: print("Calculating unnormalized full-sky kappa...")
    lcltt = theory.lCl('TT',range(lmax))
    with bench.show("reconstruction"):
        ukappa_alm = qe.qe_tt_simple(Xmap,lcltt=lcltt,nltt_deconvolved=0.,lmin=2,lmax=lmax)
    del Xmap

    # alms of input kappa
    ikappa = enmap.read_map(ksim_location+"phi_%s/kappaMap_%s.fits" % (sindex,sindex))
    ikappa.wcs = wcs
    assert ikappa.shape==shape
    ik_alm = cs.map2alm(ikappa,lmax=mlmax).astype(np.complex128)
    del ikappa
    kappa_alm = hp.almxfl(ukappa_alm,Al).astype(np.complex128)
    del ukappa_alm

    # cross and auto powers
    cri = hp.alm2cl(ik_alm,kappa_alm)
    crr = hp.alm2cl(kappa_alm)
    cii = hp.alm2cl(ik_alm)

    del kappa_alm
    del ik_alm

    ls = np.arange(len(crr))

    cri[ls<2] = np.nan
    crr[ls<2] = np.nan
    cii[ls<2] = np.nan


    cents,bri = binner.binned(ls,cri)
    cents,brr = binner.binned(ls,crr)
    cents,bii = binner.binned(ls,cii)


    mstats.add_to_stats("ri",bri)
    mstats.add_to_stats("rr",bri)
    mstats.add_to_stats("ii",bri)
    mstats.add_to_stats("diff",(bri-bii)/bii)

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

mstats.get_stats()

if rank==0:

    bri = mstats.stats['ri']['mean']
    brr = mstats.stats['rr']['mean']
    bii = mstats.stats['ii']['mean']
    diff = mstats.stats['diff']['mean']

    ebri = mstats.stats['ri']['errmean']
    ebrr = mstats.stats['rr']['errmean']
    ebii = mstats.stats['ii']['errmean']
    ediff = mstats.stats['diff']['errmean']

    # plot
    pl = io.Plotter(yscale='log',xscale='log')
    pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
    pl.add_err(cents,bri,yerr=ebri,ls="none",marker="o")
    pl.add_err(cents,brr,yerr=ebrr,ls="none",marker="o")
    pl.add_err(cents,bii,yerr=ebii,ls="none",marker="x")
    pl.add(lpls,lpal*(lpls*(lpls+1.))**2./4.,ls="-.")
    pl.done(io.dout_dir+"fullsky_qe_result_%d.png" % lmax)

    pl = io.Plotter()
    pl.add_err(cents,diff,yerr=ediff,ls="-",marker="o")
    pl.hline()
    pl._ax.set_ylim(-0.03,0.03)
    pl.done(io.dout_dir+"fullsky_qe_result_diff_%d.png" % lmax)
