from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from falafel.qe import symlens_norm as get_norm

thloc = "/scratch/r/rbond/msyriac/data/sims/alex/v0.4/cosmo2017_10K_acc3"
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)


ells = np.arange(3100)
uctt = tctt = theory.lCl('TT',ells)
ucee = tcee = theory.lCl('EE',ells)
ucbb = tcbb = theory.lCl('BB',ells)
ucte = tcte = theory.lCl('TE',ells)
clkk = theory.gCl('kk',ells)

ls,Als,al_mv_pol,al_mv,Al_te_hdv = get_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=100,lmax=3000,plot=False)
io.save_cols("norm.txt",(ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv))
