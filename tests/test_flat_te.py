from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,curvedsky as cs,lensing
from enlib import bench
import numpy as np
import os,sys
import healpy as hp
from falafel import qe
import symlens

"""
What's up with curved sky TE?
Let's test it by doing flat sky TE.

Hmm, looks like the norm is pretty much correct for flat sky TE.
"""


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("--nsims",     type=int,  default=5,help="Number of sims. If not specified, runs on data.")
parser.add_argument("--res",     type=float,  default=1.5,help="Resolution in arcminutes.")
parser.add_argument("--width",     type=float,  default=10.,help="Width in degrees.")
parser.add_argument("--tapwidth",     type=float,  default=1.5,help="Width in degrees.")
parser.add_argument("--lmaxt",     type=int,  default=3000,help="help")
parser.add_argument("--lmaxp",     type=int,  default=5000,help="help")
parser.add_argument("--lmint",     type=int,  default=100,help="help")
parser.add_argument("--lminp",     type=int,  default=100,help="help")
parser.add_argument("--sim-location",     type=str,  default=None,help="Path to sims.")
parser.add_argument("--norm-file",     type=str,  default="norm.txt",help="Norm file.")
parser.add_argument("--dtype",     type=int,  default=64,help="dtype bits")
args = parser.parse_args()
dtype = np.complex128 if args.dtype==64 else np.complex64

# Resolution
res = args.res
shape,wcs = maps.rect_geometry(width_deg=args.width,px_res_arcmin=res,proj="car")
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

# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = args.nsims
num_each,each_tasks = mpi.mpi_distribute(Nsims,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
s = stats.Stats(comm)


# Theory
thloc = sim_location + "cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

# Norm
Als = {}
ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],Als['mvpol'],Als['mv'] = np.loadtxt(args.norm_file,unpack=True)

bin_edges = np.arange(80,3000,80)
modlmap = enmap.modlmap(shape,wcs)    
binner = stats.bin2D(modlmap,bin_edges)
cents = binner.centers
power = lambda x,y: binner.bin(np.real(x*y.conj()))[1]

norm = maps.interp(ls,Als['TE'])(modlmap)
fc = maps.FourierCalc(shape,wcs)
for task in my_tasks:

    # Load sims
    sindex = str(task).zfill(5)
    alm = maps.change_alm_lmax(hp.read_alm(sim_location+"fullskyLensedUnabberatedCMB_alm_set00_%s.fits" % sindex,hdu=(1,2,3)),mlmax).astype(dtype)

    imap = cs.alm2map(alm,enmap.zeros((3,)+shape[-2:],wcs),spin=0)
    taper,w2 = maps.get_taper_deg(shape,wcs,args.tapwidth)
    w3 = np.mean(taper**3.)
    
    imap = imap * taper
    _,kmap,_ = fc.power2d(imap,rot=False)

    # Inputs
    ikalm = lensing.phi_to_kappa(maps.change_alm_lmax(hp.read_alm(sim_location+"fullskyPhi_alm_%s.fits" % (sindex)),mlmax))
    ikmap = cs.alm2map(ikalm,enmap.zeros((1,)+shape[-2:],wcs),spin=0)
    ikmap = ikmap * taper
    _,kikmap,_ = fc.power2d(ikmap)
    
    tmask = maps.mask_kspace(shape,wcs,lmin=args.lmint,lmax=args.lmaxt)
    feed_dict = {
        'X': kmap[0],
        'Y': kmap[1],
        'uC_T_E':theory.lCl('TE',modlmap),
        'uC_T_T':theory.lCl('TT',modlmap),
        'uC_E_E':theory.lCl('EE',modlmap),
        'tC_T_T':theory.lCl('TT',modlmap),
        'tC_T_E':theory.lCl('TE',modlmap),
        'tC_E_E':theory.lCl('EE',modlmap),
    }
    krecon = norm * symlens.unnormalized_quadratic_estimator(shape,wcs,feed_dict,"hu_ok",'TE',xmask=tmask,ymask=tmask,pixel_units=True)

    s.add_to_stats("ri",power(krecon,kikmap)/w3)
    s.add_to_stats("ii",power(kikmap,kikmap)/w2)
    
    

    if rank==0: print ("Rank 0 done with task ", task+1, " / " , len(my_tasks))
s.get_stats()

if rank==0:

    ii = s.stats['ii']['mean']
    ri = s.stats['ri']['mean']
    eri = s.stats['ri']['errmean']

    pl = io.Plotter(xyscale='linlog',xlabel='$L$',ylabel='$\\Delta C_L / C_L$')
    pl.add(cents,ii)
    pl.add(cents,ri)
    pl.hline()
    # pl._ax.set_ylim(-0.5,0.5)
    pl.done(os.environ['WORK']+'/verify_TE.png')

    pl = io.Plotter(xyscale='linlin',xlabel='$L$',ylabel='$\\Delta C_L / C_L$')
    pl.add_err(cents,(ri-ii)/ii,yerr=eri/ii)
    pl.hline()
    pl._ax.set_ylim(-0.1,0.1)
    pl.done(os.environ['WORK']+'/verify_TE_diff.png')
    
