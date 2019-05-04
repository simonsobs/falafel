from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs,lensing
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe

"""
We will verify lensing reconstruction for the following estimators:
T-only: TT
E-only: EE
P-only: EE + EB
MV: TT + EE + TE + EB + TB

by:
1. comparing recon x input with input x input
2. comparing recon x recon with input x input + N_L from A_L
3. comparing curl x input with zero

We do this for 1uK-arcmin noise and 1 arcmin beam,
lmin = 300, lmax_T = 3000, lmax_P = 5000,
on the full-sky with no mask.

"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("--nsims",     type=int,  default=5,help="Number of sims. If not specified, runs on data.")
parser.add_argument("--res",     type=float,  default=1.5,help="Resolution in arcminutes.")
parser.add_argument("--lmaxt",     type=int,  default=3000,help="help")
parser.add_argument("--lmaxp",     type=int,  default=5000,help="help")
parser.add_argument("--lmint",     type=int,  default=100,help="help")
parser.add_argument("--lminp",     type=int,  default=100,help="help")
parser.add_argument("--beam",     type=float,  default=1.,help="Beam in arcminutes.")
parser.add_argument("--noise",     type=float,  default=1.,help="Noise in muK-arcmin.")
parser.add_argument("--sim-location",     type=str,  default=None,help="Path to sims.")
parser.add_argument("--norm-file",     type=str,  default="norm.txt",help="Norm file.")
parser.add_argument("--dtype",     type=int,  default=32,help="dtype bits")
args = parser.parse_args()
dtype = np.complex128 if args.dtype==64 else np.complex64

# Resolution
res = args.res
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj="car")
mlmax = max(args.lmaxt,args.lmaxp) + 500

# Sim location
try:
    if args.sim_location is not None: raise
    from soapack import interfaces as sints
    sim_location = sints.dconfig['actsims']['signal_path']
except:
    sim_location = args.sim_location
    assert sim_location is not None

# Beam and noise
ells = np.arange(mlmax)
lbeam = maps.gauss_beam(ells,args.beam)
ps_noise = np.zeros((3,3,ells.size))
ps_noise[0,0] = (args.noise*np.pi/180./60.)**2.
ps_noise[1,1] = (args.noise*np.pi/180./60.)**2. * 2.
ps_noise[2,2] = (args.noise*np.pi/180./60.)**2. * 2.

# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = args.nsims
num_each,each_tasks = mpi.mpi_distribute(Nsims,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
s = stats.Stats(comm)

# Reduction
malm2cl = lambda x,y: hp.alm2cl(x,y)
def reduce(name,irecon,ialm):
    orecon = qe.almxfl(irecon,Als[name])
    s.add_to_stats(name+"_gauto",malm2cl(orecon[0],orecon[0]))
    s.add_to_stats(name+"_cauto",malm2cl(orecon[1],orecon[1]))
    s.add_to_stats(name+"_gcross",malm2cl(orecon[0],ialm))
    s.add_to_stats(name+"_ccross",malm2cl(orecon[1],ialm))

# Theory
thloc = sim_location + "cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

# Norm
Als = {}
ls,Als['tonly'],Als['eonly'],Als_eb,Als_te,Als_tb,Als['ponly'],Als['mv'] = np.loadtxt(args.norm_file,unpack=True)
    
for task in my_tasks:

    # Load sims
    sindex = str(task).zfill(5)
    alm = maps.change_alm_lmax(hp.read_alm(sim_location+"fullskyLensedUnabberatedCMB_alm_set00_%s.fits" % sindex,hdu=(1,2,3)),mlmax).astype(dtype)

    # Add beam deconvolved noise
    alm = alm + np.nan_to_num(qe.almxfl(cs.rand_alm_healpy(ps_noise, lmax=mlmax, seed=(100,200,task), dtype=dtype),1./lbeam))
    ntt = maps.interp(ells,np.nan_to_num(ps_noise[0,0]/lbeam**2.))
    npp = maps.interp(ells,np.nan_to_num(ps_noise[1,1]/lbeam**2.))

    # Filter
    talm_x = qe.filter_alms(alm[0],lambda x: theory.lCl('TT',x)/(theory.lCl('TT',x)+ntt(x)),args.lmint,args.lmaxt)
    ealm_x = qe.filter_alms(alm[1],lambda x: theory.lCl('EE',x)/(theory.lCl('EE',x)+npp(x)),args.lminp,args.lmaxp)
    balm_x = qe.filter_alms(alm[2],lambda x: theory.lCl('BB',x)/(theory.lCl('BB',x)+npp(x)),args.lminp,args.lmaxp)
    talm_y = qe.filter_alms(alm[0],lambda x: 1./(theory.lCl('TT',x)+ntt(x)),args.lmint,args.lmaxt)
    ealm_y = qe.filter_alms(alm[1],lambda x: 1./(theory.lCl('EE',x)+npp(x)),args.lminp,args.lmaxp)
    balm_y = qe.filter_alms(alm[2],lambda x: 1./(theory.lCl('BB',x)+npp(x)),args.lminp,args.lmaxp)
    del alm
    
    # Inputs
    ikalm = lensing.phi_to_kappa(maps.change_alm_lmax(hp.read_alm(sim_location+"fullskyPhi_alm_%s.fits" % (sindex)),mlmax))
    
    # Recons

    with bench.show("recon mv"):
        recon_mv,dmap_t,dmap_p = qe.qe_mv(shape,wcs,talm_x,ealm_x,balm_x,talm_y,ealm_y,balm_y,args.lmaxt,args.lmaxp,mlmax)

    reduce("mv",recon_mv,ikalm)
    del recon_mv
    
    with bench.show("recon tt"):
        recon_tonly = qe.deflection_map_to_kappa_curl_alms(dmap_t,mlmax)

    reduce("tonly",recon_tonly,ikalm)
    del recon_tonly,talm_x,talm_y,dmap_t
        
    with bench.show("recon pol"):
        recon_ponly = qe.deflection_map_to_kappa_curl_alms(dmap_p,mlmax)

    reduce("ponly",recon_ponly,ikalm)
    del recon_ponly, balm_x, balm_y,dmap_p
        
    with bench.show("recon eonly"):
        recon_eonly = qe.qe_pol_only(shape,wcs,ealm_x,ealm_x*0,ealm_y,ealm_y*0,args.lmaxp,mlmax)

    reduce("eonly",recon_eonly,ikalm)
    del recon_eonly, ealm_x, ealm_y
        
    s.add_to_stats("iauto",malm2cl(ikalm,ikalm))
    del ikalm

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))
    
s.get_stats()
    

        
if rank==0:

    # Get data
    icls = s.stats['iauto']['mean']
    ells = np.arange(len(icls))
    combs = ['tonly','ponly','eonly','mv']
    ostats = ['gauto','cauto','gcross','ccross']
    cls = {}
    for comb in combs:
        cls[comb] = {}
        for stat in ostats:
            cls[comb][stat] = {}
            cls[comb][stat]['mean'] = s.stats[comb+"_"+stat]['mean']
            cls[comb][stat]['err'] = s.stats[comb+"_"+stat]['errmean']
        
    
    # Make plots
    for comb in combs:
        # grad
        pl = io.Plotter(xyscale='loglog',xlabel='$L$',ylabel='$C_L$')
        fells = np.arange(2,mlmax)
        pl.add(fells,theory.gCl('kk',fells),lw=3)
        pl.add(ells,icls,color='k',alpha=0.5)
        pl.add(ls,maps.interp(ells,icls)(ls) + (Als[comb]*ls*(ls+1)/4.))
        pl.add_err(ells,cls[comb]['gauto']['mean'],yerr=cls[comb]['gauto']['err'])
        pl.add_err(ells,cls[comb]['gcross']['mean'],yerr=cls[comb]['gauto']['err'])
        pl._ax.set_ylim(1e-9,1e-5)
        pl.done('verify_grad_%s.png' % comb)

        # grad diff
        pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$\\Delta C_L / C_L$')
        pl.add_err(ells,(cls[comb]['gcross']['mean']-icls)/icls,yerr=cls[comb]['gcross']['err']/icls)
        pl.hline()
        pl._ax.set_ylim(-0.5,0.5)
        pl.done('verify_grad_diff_%s.png' % comb)
        
        # curl
        pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$C_L$')
        pl.add_err(ells,cls[comb]['ccross']['mean'],yerr=cls[comb]['cauto']['err'])
        pl.hline()
        pl.done('verify_curl_%s.png' % comb)
        


    
