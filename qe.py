from enlib import curvedsky as cs, enmap, lensing as enlensing, powspec
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent
from orphics import io,cosmology,lensing,maps

def isotropic_filter_T(imap=None,alm=None,lcltt=None,ucltt=None,
                       nltt_deconvolved=None,tcltt=None,lmin=None,lmax=None,gradient=False):
    if alm is None: alm = cs.map2alm(imap,lmax=lmax)
    if gradient:
        if ucltt is None: ucltt = lcltt
        numer = ucltt
    else:
        numer = 1.
    denom = tcltt if tcltt is not None else lcltt+nltt_deconvolved
    wfilter = np.nan_to_num(numer/denom)
    ells = np.arange(0,wfilter.size,1)
    if lmin is not None: wfilter[ells<lmin] = 0
    wfilter[ells>lmax] = 0
    return hp.almxfl(alm,wfilter)


def gradient_T_map(shape,wcs,alm):
    """
    Given appropriately Wiener filtered temperature map alms,
    returns a real-space map containing the gradient of T.
    """
    omap = enmap.zeros((2,)+shape[-2:],wcs)
    return cs.alm2map(alm,omap,deriv=True) # note that deriv=True gives the scalar derivative of a scalar alm field

def gradient_E_map(alm):
    """
    Given appropriately Wiener filtered E-mode alms,
    returns a real-space map containing the gradient of E.
    """
    pass

def gradient_B_map(alm):
    """
    Given appropriately Wiener filtered E-mode alms,
    returns a real-space map containing the gradient of B.
    """
    pass


def qe_tt_simple(Xmap,Ymap=None,lcltt=None,ucltt=None, \
                 nltt_deconvolved=None,tcltt=None,nltt_deconvolved_y=None,tcltt_y=None,
                 lmin=None,lmax=None,lmin_y=None,lmax_y=None,do_curl=False):

    if Ymap is None: assert (nltt_deconvolved_y is None) and (tcltt_y is None) and (lmin_y is None) and (lmax_y is None)
    if nltt_deconvolved_y is None: nltt_deconvolved_y = nltt_deconvolved
    if tcltt_y is None: tcltt_y = tcltt
    if lmin_y is None: lmin_y = lmin
    if lmax_y is None: lmax_y = lmax

    iXalm = cs.map2alm(Xmap,lmax=lmax)
    if Ymap is None:
        iYalm = iXalm.copy()
    else:
        iYalm = cs.map2alm(Ymap,lmax=lmax)
        
    Xalm = isotropic_filter_T(alm=iXalm,lcltt=lcltt,ucltt=ucltt,
                              nltt_deconvolved=nltt_deconvolved,tcltt=tcltt,lmin=lmin,lmax=lmax,gradient=True)
    Yalm = isotropic_filter_T(alm=iYalm,lcltt=lcltt,ucltt=ucltt,
                              nltt_deconvolved=nltt_deconvolved_y,tcltt=tcltt_y,lmin=lmin_y,lmax=lmax_y,gradient=False)
    shape,wcs = Xmap.shape,Xmap.wcs
    return qe_tt(shape,wcs,Xalm,Yalm,lmax=lmax,do_curl=do_curl)
    
def qe_tt(shape,wcs,Xalm,Yalm,do_curl=False,lmax=None):

    gradT = gradient_T_map(shape,wcs,Xalm)
    highT = enmap.zeros(shape[-2:],wcs)
    highT = cs.alm2map(Yalm,highT)

    px = gradT[0] * highT
    py = gradT[1] * highT
    
    alm_px = cs.map2alm(px,lmax=lmax)
    alm_py = cs.map2alm(py,lmax=lmax)
    
    dpx = enmap.zeros((2,)+shape[-2:],wcs)
    dpx = cs.alm2map(alm_px,dpx,deriv=True)
    dpxdx = dpx[0].copy()
    if do_curl: dpxdy = dpx[1].copy()
    dpy = enmap.zeros((2,)+shape[-2:],wcs)
    dpy = cs.alm2map(alm_py,dpy,deriv=True)
    if do_curl: dpydx = dpy[0].copy()
    dpydy = dpy[1].copy()

    phi = dpxdx + dpydy
    alm_phi = cs.map2alm(phi,lmax=lmax)
    if do_curl:
        curl = dpydx - dpxdy
        alm_curl = cs.map2alm(curl,lmax=lmax)
        return alm_phi,alm_curl
    else:
        return alm_phi
    


res = 8.0
lmax = 500
dtype = np.float32


theory = cosmology.loadTheorySpectraFromCAMB("data/Aug6_highAcc_CDM",get_dimensionless=False)

ells = np.arange(0,lmax,1)
ntt = nee = nbb = ells*0.
ellmin_t = 2
ellmax_t = lmax
bin_edges = np.linspace(2,lmax,40)
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
                                             width_deg=10.,px_res_arcmin=res/2.)

Nl = maps.interp(ls,nlkks['TT'])(ells)

# pl = io.Plotter(yscale='log')
# pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
# pl.add(ells,Nl,ls="--")
# pl.done()




shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.))
pfile = "data/Aug6_highAcc_CDM_lenspotentialCls.dat"
ps = powspec.read_camb_full_lens(pfile).astype(dtype)
print("Making random lensed map...")
Xmap,ikappa, = enlensing.rand_map(shape[-2:], wcs, ps, lmax=lmax,dtype=dtype,output="lk")
# io.plot_img(Xmap)
print("Calculating unnormalized phi...")
lcltt = theory.lCl('TT',range(lmax))
uphi_alm = qe_tt_simple(Xmap,lcltt=lcltt,nltt_deconvolved=0.,lmin=2,lmax=lmax)

Al = Nl
ik_alm = cs.map2alm(ikappa,lmax=lmax).astype(np.complex128)
kappa_alm = hp.almxfl(uphi_alm,Al).astype(np.complex128)

cri = hp.alm2cl(ik_alm,kappa_alm)
crr = hp.alm2cl(kappa_alm)


pl = io.Plotter(yscale='log')
pl.add(ells,theory.gCl('kk',ells),lw=3,color='k')
pl.add(ells,Nl,ls="--")
pl.add(ells,cri[:-1])
pl.add(ells,crr[:-1])
pl._ax.set_ylim(1e-10,1e-6)
pl.done()
