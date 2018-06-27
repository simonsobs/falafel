from enlib import curvedsky as cs
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent 

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
    wfilter[ells<lmax] = 0
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
                 nltt_deconvolved=None,tcltt=None,nltt_deconvolved_y=None,tcltt_y=None,lmin=None,lmax=None,lmin_y=None,lmax_y=None):

    if nltt_deconvolved_y is None: nltt_deconvolved_y = nltt_deconvolved
    if tcltt_y is None: tcltt_y = tcltt
    if lmin_y is None: lmin_y = lmin
    if lmax_y is None: lmax_y = lmax

    iXalm = cs.map2alm(imap,lmax=lmax)
    if Ymap is None:
        assert nltt_deconvolved_y is None and tcltt_y is None and lmin_y is None and lmax_y is None
        iYalm = iXalm.copy()
    Xalm = isotropic_filter_T(alm=iXalm,lcltt=lcltt,ucltt=ucltt,
                              nltt_deconvolved=nltt_deconvolved,tcltt=tcltt,lmin=lmin,lmax=lmax,gradient=True)
    Yalm = isotropic_filter_T(alm=iYalm,lcltt=lcltt,ucltt=ucltt,
                              nltt_deconvolved=nltt_deconvolved_y,tcltt=tcltt_y,lmin=lmin_y,lmax=lmax_y,gradient=False)
    shape,wcs = Xmap.shape,Xmap.wcs
    return qe_tt(shape,wcs,Xalm,Yalm)
    
def qe_tt(shape,wcs,Xalm,Yalm):

    gradT = gradient_T_map(shape,wcs,Xalm)
    highT = cs.alm2map(Yalm)

    px = gradT[0] * highT
    py = gradT[1] * highT
    
    alm_px = cs.map2alm(px)
    alm_py = cs.map2alm(py)
    
    omap = enmap.zeros((2,)+shape[-2:],wcs)
    dpx = cs.alm2map(alm_px,omap,deriv=True)
    dpxdx = omap[0].copy()
    dpxdy = omap[1].copy()
    omap = enmap.zeros((2,)+shape[-2:],wcs)
    omap = cs.alm2map(alm_py,omap,deriv=True)
    dpydx = omap[0].copy()
    dpydy = omap[1].copy()

    phi = dpxdx + dpydy
    curl = dpydx - dpxdy

    return phi,curl
    


        
