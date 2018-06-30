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


mlmax = lmax + 250 # lmax used for harmonic transforms



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

sim_location = "/gpfs01/astro/workarea/msyriac/data/sims/msyriac/lenspix/"
ksim_location = "/gpfs01/astro/workarea/msyriac/data/sims/msyriac/lenspix/"


bin_edges = np.linspace(2,lmax,300)
binner = stats.bin1D(bin_edges)

mstats = stats.Stats(comm)

lstr = ""
if lmax==2000: lstr = "_lmax2250"

for task in my_tasks:


    sim_index = task

    sindex = str(sim_index)
    if rank==0: print("Loading lensed map...")
    try:
        with io.nostdout():
            Xmap = hp.read_map(sim_location+"cosmo2017_test%s_lmax%d_nside2048_interp1.51_lensed_%s.fits" % (lstr,lmax+250,sindex))
    except:
        print("Missing lensed map for ", task)
        continue


    ### DO FULL SKY RECONSTRUCTION
    if rank==0: print("Calculating unnormalized full-sky kappa...")
    lcltt = theory.lCl('TT',range(lmax))
    with bench.show("reconstruction"):
        ukappa_alm = qe.qe_tt_simple(Xmap,lcltt=lcltt,nltt_deconvolved=0.,lmin=2,lmax=lmax,healpix=True)
    del Xmap
    # alms of falafel reconstruction
    kappa_alm = hp.almxfl(ukappa_alm,Al).astype(np.complex128)
    del ukappa_alm
    

    # alms of input kappa
    try:
        with io.nostdout():
            iphi = hp.read_map(ksim_location+"cosmo2017_test%s_lmax%d_nside2048_interp1.51_phimap_%s.fits" % (lstr,lmax+250,sindex))
    except:
        print("Missing phi map for ", task)
        continue
        
    p2k = ells*(ells+1.)/2.
    ik_alm = hp.almxfl(qe.gmap2alm(iphi,lmax=mlmax,healpix=True).astype(np.complex128),p2k)
    del iphi


    # alms of LensPix reconstruction
    try:
        with io.nostdout():
            iphir = hp.read_map(ksim_location+"cosmo2017_test%s_lmax%d_nside2048_interp1.51_phiTT_map_%s.fits" % (lstr,lmax+250,sindex))
    except:
        print("Missing lenspix reconstruction map for ", task)
        continue
    rk_alm = hp.almxfl(qe.gmap2alm(iphir,lmax=mlmax,healpix=True).astype(np.complex128),p2k)
    del iphir
    

    # cross and auto powers
    cri = hp.alm2cl(ik_alm,kappa_alm)
    crr = hp.alm2cl(kappa_alm)
    cii = hp.alm2cl(ik_alm)
    chi = hp.alm2cl(ik_alm,rk_alm)
    chh = hp.alm2cl(rk_alm)

    del kappa_alm
    del ik_alm
    del rk_alm

    ls = np.arange(len(crr))

    cri[ls<2] = np.nan
    crr[ls<2] = np.nan
    chi[ls<2] = np.nan
    chh[ls<2] = np.nan
    cii[ls<2] = np.nan


    cents,bri = binner.binned(ls,cri)
    cents,brr = binner.binned(ls,crr)
    cents,bhi = binner.binned(ls,chi)
    cents,bhh = binner.binned(ls,chh)
    cents,bii = binner.binned(ls,cii)


    mstats.add_to_stats("ri",bri)
    mstats.add_to_stats("rr",brr)
    mstats.add_to_stats("hi",bhi)
    mstats.add_to_stats("hh",bhh)
    mstats.add_to_stats("ii",bii)
    mstats.add_to_stats("diff",(bri-bii)/bii)
    mstats.add_to_stats("hdiff",(bhi-bii)/bii)

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

mstats.get_stats()

if rank==0:

    bri = mstats.stats['ri']['mean']
    brr = mstats.stats['rr']['mean']
    bhi = mstats.stats['hi']['mean']
    bhh = mstats.stats['hh']['mean']
    bii = mstats.stats['ii']['mean']
    diff = mstats.stats['diff']['mean']
    hdiff = mstats.stats['hdiff']['mean']

    ebri = mstats.stats['ri']['errmean']
    ebrr = mstats.stats['rr']['errmean']
    ebhi = mstats.stats['hi']['errmean']
    ebhh = mstats.stats['hh']['errmean']
    ebii = mstats.stats['ii']['errmean']
    ediff = mstats.stats['diff']['errmean']
    ehdiff = mstats.stats['hdiff']['errmean']

    # plot
    pl = io.Plotter(yscale='log',xscale='log')
    pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
    pl.add_err(cents,bri,yerr=ebri,ls="none",marker="o",label='falfel hpix rxi lmax=%d' % lmax)
    pl.add_err(cents,brr,yerr=ebrr,ls="none",marker="o",label='falfel hpix rxr lmax=%d' % lmax)
    pl.add_err(cents,bhi,yerr=ebhi,ls="none",marker="o",label='lenspix hpix rxi lmax=%d' % (lmax+250))
    pl.add_err(cents,bhh,yerr=ebhh,ls="none",marker="o",label='lenspix hpix rxr lmax=%d' % (lmax+250))
    pl.add_err(cents,bii,yerr=ebii,ls="none",marker="x",label='hpix ixi')
    pl.add(lpls,lpal*(lpls*(lpls+1.))**2./4.,ls="-.")
    pl.legend()
    pl.done(io.dout_dir+"fullsky_qe_result_%d_hpix.png" % lmax)

    pl = io.Plotter()
    pl.add_err(cents,diff,yerr=ediff,ls="-",marker="o",label='falafel on hpix lmax=%d' % lmax)
    pl.add_err(cents,hdiff,yerr=ehdiff,ls="-",marker="o",label='lenspix on hpix lmax=%d' % (lmax+250))
    pl.hline()
    pl.legend()
    pl._ax.set_ylim(-0.2,0.2)
    pl._ax.set_xlim(0,lmax)
    pl.done(io.dout_dir+"fullsky_qe_result_diff_%d_hpix.png" % lmax)
