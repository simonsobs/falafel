from __future__ import print_function
from enlib import curvedsky as cs, enmap, lensing as enlensing, powspec, bench
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent
from orphics import io,cosmology,lensing,maps,stats,mpi
from falafel import qe
import os,sys


lmax = int(sys.argv[1]) # cmb ellmax
#lmax = 1000 #int(sys.argv[1]) # cmb ellmax
Nsims = int(sys.argv[2])


comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]


res = 1.0 # resolution in arcminutes
#mlmax = 2*lmax #+ 250 # lmax used for harmonic transforms
mlmax = lmax + 250 # lmax used for harmonic transforms
# mlmax = 2*lmax # lmax used for harmonic transforms



# load theory
camb_root = "data/cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(camb_root,get_dimensionless=False)

# ells corresponding to modes in the alms
ells = np.arange(0,mlmax,1)
    
#lpls,lpal = np.loadtxt("data/nls_%d.txt" % lmax,unpack=True)
lpfile = "/gpfs01/astro/workarea/msyriac/data/sims/msyriac/lenspix/cosmo2017_lmax_fix_lens_lmax_%d_qest_lmax_%d_AL.txt" % (lmax+2000,lmax)
#lpfile = "/gpfs01/astro/workarea/msyriac/data/sims/msyriac/lenspix/cosmo2017_lmax_fix_lens_lmax_%d_qest_lmax_%d_AL.txt" % (2000+2000,2000)
lpls,lpal = np.loadtxt(lpfile,unpack=True,usecols=[0,1])
lpal = lpal / (lpls) / (lpls+1.)

dh_nls = np.nan_to_num(lpal*(lpls*(lpls+1.))**2./4.)
dh_als = np.nan_to_num(dh_nls * 2. / lpls /(lpls+1))
Al = dh_als
Al = maps.interp(lpls,dh_als)(ells)
Nl = maps.interp(lpls,dh_nls)(ells)
Al[ells<2] = 0
Nl[ells<2] = 0

shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
#sim_location = "/global/cscratch1/sd/engelen/simsS1516_v0.3/data/"
#ksim_location = "/global/cscratch1/sd/dwhan89/shared/act/simsS1516_v0.3/data/"

sim_location = "/gpfs01/astro/workarea/msyriac/data/sims/alex/v0.3/"
ksim_location = "/gpfs01/astro/workarea/msyriac/data/sims/alex/v0.3/"


bin_edges = np.logspace(np.log10(2),np.log10(lmax),100)
#bin_edges = np.linspace(2,lmax,300)
binner = stats.bin1D(bin_edges)

mstats = stats.Stats(comm)

for task in my_tasks:


    sim_index = task+1
    assert sim_index>0

    sindex = str(sim_index).zfill(5)
    if rank==0: print("Loading lensed map...")
    #Xmap = enmap.read_map(sim_location+"cmb_set00_%s/fullskyLensedMapUnaberrated_T_%s.fits" % (sindex,sindex))
    Xmap = enmap.read_map(sim_location+"fullskyLensedMapUnaberrated_T_%s.fits" % (sindex))
    Xmap = enmap.enmap(Xmap,wcs)
    assert Xmap.shape==shape
    # xalm = cs.map2alm(Xmap-Xmap.mean(),lmax=mlmax).astype(np.complex128)
    # gy,gx = qe.gradient_T_map(shape,wcs,xalm)
    # gy_alm = cs.map2alm(gy-gy.mean(),lmax=mlmax).astype(np.complex128)
    # gx_alm = cs.map2alm(gx-gx.mean(),lmax=mlmax).astype(np.complex128)
    # dyy,dyx = qe.gradient_T_map(shape,wcs,gy_alm)
    # dxy,dxx = qe.gradient_T_map(shape,wcs,gx_alm)
    # div = dyy + dxx
    # dalm = cs.map2alm(div-div.mean(),lmax=mlmax).astype(np.complex128)
    
    # cls = hp.alm2cl(dalm)
    # mstats.add_to_stats("cls",cls)
    # continue
        


    ### DO FULL SKY RECONSTRUCTION
    if rank==0: print("Calculating unnormalized full-sky kappa...")
    lcltt = theory.lCl('TT',range(lmax))
    ukappa_alm = qe.qe_tt_simple(Xmap,lcltt=lcltt,nltt_deconvolved=0.,lmin=2,lmax=lmax)
    # ukappa_alm = qe.qe_tt_simple(Xmap,lcltt=lcltt,nltt_deconvolved=0.,lmin=2,lmax=1000)  # !!!!!
    del Xmap
    kappa_alm = hp.almxfl(ukappa_alm,Al).astype(np.complex128)
    #del ukappa_alm
    if task==0:
        ls = np.arange(lmax)
        fls = np.ones(ls.size)
        fls[ls>100] = 0
        rkappa = enmap.zeros(shape[-2:],wcs)
        rkappa = cs.alm2map(hp.almxfl(kappa_alm,fls),rkappa,method="cyl")
        io.plot_img(rkappa,io.dout_dir+"rkappa.png")
        # io.plot_img(rkappa,io.dout_dir+"rkappa_high.png",high_res=True)
        rkappa = enmap.zeros(shape[-2:],wcs)
        rkappa = cs.alm2map(hp.almxfl(ukappa_alm,fls),rkappa,method="cyl")
        io.plot_img(rkappa,io.dout_dir+"urkappa.png")
        # io.plot_img(rkappa,io.dout_dir+"urkappa_high.png",high_res=True)
    

    # alms of input kappa
    #ikappa = enmap.read_map(ksim_location+"phi_%s/kappaMap_%s.fits" % (sindex,sindex))
    ikappa = enmap.read_map(ksim_location+"kappaMap_%s.fits" % (sindex))
    ikappa.wcs = wcs
    assert ikappa.shape==shape
    ik_alm = cs.map2alm(ikappa-ikappa.mean(),lmax=mlmax).astype(np.complex128)
    del ikappa

    # cross and auto powers
    cuu = hp.alm2cl(ukappa_alm)
    cui = hp.alm2cl(ik_alm,ukappa_alm)
    cri = hp.alm2cl(ik_alm,kappa_alm)
    crr = hp.alm2cl(kappa_alm)
    cii = hp.alm2cl(ik_alm)

    del kappa_alm
    del ik_alm

    ls = np.arange(len(crr))

    cuu[ls<2] = np.nan
    cui[ls<2] = np.nan
    cri[ls<2] = np.nan
    crr[ls<2] = np.nan
    cii[ls<2] = np.nan


    cents,buu = binner.binned(ls,cuu)
    cents,bui = binner.binned(ls,cui)
    cents,bri = binner.binned(ls,cri)
    cents,brr = binner.binned(ls,crr)
    cents,bii = binner.binned(ls,cii)

    mstats.add_to_stats("cuu",cuu)

    mstats.add_to_stats("uu",buu)
    mstats.add_to_stats("ui",bui)
    mstats.add_to_stats("ri",bri)
    mstats.add_to_stats("rr",brr)
    mstats.add_to_stats("ii",bii)
    mstats.add_to_stats("diff",(bri-bii)/bii)

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))

mstats.get_stats()

if rank==0:

    # ls = np.arange(len(cls))
    # lcltt = theory.lCl('TT',ls)
    # pl = io.Plotter(yscale='log',xscale='log')
    # pl.add(ls,(ls*(ls+1.))**2*lcltt,color='k',lw=3)
    # pl.add(ls,mstats.stats['cls']['mean'],alpha=0.5,marker="o")
    # pl.done(io.dout_dir+"cltt.png")
    # sys.exit()
    
    buu = mstats.stats['uu']['mean']
    bri = mstats.stats['ri']['mean']
    brr = mstats.stats['rr']['mean']
    bii = mstats.stats['ii']['mean']
    diff = mstats.stats['diff']['mean']

    ebuu = mstats.stats['uu']['errmean']
    ebri = mstats.stats['ri']['errmean']
    ebrr = mstats.stats['rr']['errmean']
    ebii = mstats.stats['ii']['errmean']
    ediff = mstats.stats['diff']['errmean']

    pl = io.Plotter(yscale='log',xscale='log')
    pl.add(ells,(theory.gCl('kk',ells)+Nl)/Al**2. ,lw=3,color='k')
    pl.add(ls,mstats.stats['cuu']['mean'],marker="o",label='uxu')
    vsize = mstats.vectors['cuu'].shape[0]
    for i in range(vsize):
        v = mstats.vectors['cuu'][i,:]
        pl.add(ls,v,label='uxu',alpha=0.2,color='C1')
        
    pl.legend()
    pl.done(io.dout_dir+"qe_debug.png")

    
    # plot
    pl = io.Plotter(yscale='log',xscale='log')
    pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
    pl.add_err(cents,bri,yerr=ebri,ls="none",marker="o",label='rxi')
    pl.add_err(cents,brr,yerr=ebrr,ls="none",marker="o",label='rxr')
    pl.add_err(cents,bii,yerr=ebii,ls="none",marker="x",label='ixi')
    pl.add(lpls,(lpal*(lpls*(lpls+1.))**2./4.)+theory.gCl('kk',lpls),ls="--")
    pl.add(lpls,lpal*(lpls*(lpls+1.))**2./4.,ls="-.")
    pl.legend()
    pl.done(io.dout_dir+"fullsky_qe_result_%d.png" % lmax)

    pl = io.Plotter()
    pl.add_err(cents,diff,yerr=ediff,ls="-",marker="o")
    pl.hline()
    pl._ax.set_ylim(-0.03,0.03)
    pl._ax.set_xlim(0,lmax)
    pl.done(io.dout_dir+"fullsky_qe_result_diff_%d.png" % lmax)


    pl = io.Plotter(xscale='log')
    pl.add_err(cents,diff,yerr=ediff,ls="-",marker="o")
    pl.hline()
    pl._ax.set_ylim(-0.2,0.2)
    pl._ax.set_xlim(0,lmax)
    pl.done(io.dout_dir+"fullsky_qe_result_diff_%d_log.png" % lmax)
    
