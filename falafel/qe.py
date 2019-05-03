from pixell import curvedsky as cs, enmap
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent


def rot2d(fmap):
    """
    ( f0 + i f1 ) , ( f0 - i f1)
    """
    # e.g. Rotates the map outputs M+ and M- of alm2map into sM and -sM
    return np.stack((fmap[0]+fmap[1]*1j,fmap[0]-fmap[1]*1j))

def irot2d(fmap,spin):
    """
    ( f0 + (-1)^s f1 )/2 , ( f0 - (-1)^s f1 )/(2i)
    """
    # e.g. Rotates the alms +sAlm and -sAlm into inputs a+ and a- for map2alm
    return -np.stack(((fmap[0]+((-1)**spin)*fmap[1])/2.,(fmap[0]-((-1)**spin)*fmap[1])/2./1j))

def alm2map_spin(alm,spin_alm,spin_transform,omap):
    """
    Returns
    X(n) = sum_lm  alm_s1 s2_Y_lm(n)

    where s1 and s2 are different spins

    """
    ap_am = irot2d(alm,spin=spin_alm)
    res = rot2d(cs.alm2map(ap_am,omap,spin=abs(spin_transform)))
    if spin_transform>=0:
        return res
    else:
        return res[::-1,...]

def almxfl(alm,fl):
    ncomp = alm.shape[0]
    assert ncomp in [1,2,3]
    res = alm.copy()
    for i in range(ncomp): res[i] = hp.almxfl(alm[i],fl)
    return res

def pol_alms(Ealm,Balm):
    return np.stack((Ealm+1j*Balm,Ealm-1j*Balm))

def gradient_spin(shape,wcs,alm,lmax,mlmax,spin):
    """
    Given appropriately Wiener filtered temperature map alms,
    returns a real-space map containing the gradient.
    """
    omap = enmap.zeros((2,)+shape[-2:],wcs)
    ells = np.arange(0,mlmax)
    if spin==0:
        fl = np.sqrt(ells*(ells+1.))
        spin_out = 1
        sign = 1
    elif spin==(-2):
        fl = np.sqrt((ells-1)*(ells+2.))
        spin_out = -1
        sign = 1 #!!! this sign is not understood
    elif spin==2:
        fl = np.sqrt((ells-2)*(ells+3.))
        spin_out = 3
        sign = -1
    fl[ells>lmax] = 0
    salms = almxfl(alm,fl)
    return sign*alm2map_spin(salms,spin,spin_out,omap)


def deflection_map_to_kappa_curl_alms(dmap,mlmax):
    dmap[1] = dmap[0].conj() # !!! this is being set manually instead of dropping out naturally
    res = cs.map2alm(enmap.enmap(-irot2d(dmap,spin=0).real,dmap.wcs),spin=1,lmax=mlmax)
    ells = np.arange(0,mlmax)
    fl = np.sqrt(ells*(ells+1.))/2
    res = almxfl(res,fl)
    return res

def qe_spin_temperature_deflection(shape,wcs,Xalm,Yalm,lmax_x,mlmax):
    grad = gradient_spin(shape,wcs,np.stack((Xalm,Xalm)),lmax_x,mlmax,spin=0)
    ymap = cs.alm2map(Yalm,enmap.zeros(shape[-2:],wcs))
    prod = -grad*ymap
    return enmap.enmap(prod,wcs)

def qe_spin_pol_deflection(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,lmax_x,mlmax):
    palms = pol_alms(X_Ealm,X_Balm)
    grad_p2 = gradient_spin(shape,wcs,palms,lmax_x,mlmax,spin=2)
    grad_m2 = gradient_spin(shape,wcs,palms,lmax_x,mlmax,spin=-2)
    # E_alm, B_alm -> Q(n), U(n) -> Q+iU, Q-iU
    ymap = rot2d(cs.alm2map(np.stack((Y_Ealm,Y_Balm)),enmap.zeros((2,)+shape[-2:],wcs),spin=2)) #!!
    prod = -grad_m2*ymap[0]-grad_p2*ymap[1]
    return enmap.enmap(prod,wcs)/2 # !! this factor of 2 is not understood
    
def qe_temperature_only(shape,wcs,Xalm,Yalm,lmax_x,mlmax):
    dmap = qe_spin_temperature_deflection(shape,wcs,Xalm,Yalm,lmax_x,mlmax)
    return deflection_map_to_kappa_curl_alms(dmap,mlmax)

def qe_pol_only(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,lmax_x,mlmax):
    dmap = qe_spin_pol_deflection(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,lmax_x,mlmax)
    return deflection_map_to_kappa_curl_alms(dmap,mlmax)

def qe_mv(shape,wcs,X_Talm,X_Ealm,X_Balm,Y_Talm,Y_Ealm,Y_Balm,lmax_x,mlmax):
    dmap_t = qe_spin_temperature_deflection(shape,wcs,X_Talm,Y_Talm,lmax_x,mlmax)
    dmap_p = qe_spin_pol_deflection(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,lmax_x,mlmax)
    return deflection_map_to_kappa_curl_alms(dmap_t+dmap_p,mlmax)


"""
LEGACY
"""

def get_fullsky_res(npix,squeeze=0.8):
    "Slightly squeezed pixel width in radians given npix pixels on the full sky."
    return np.sqrt(4.*np.pi/npix) * squeeze



def gmap2alm(imap,lmax,healpix=False,iter=3):
    """Generic map -> alm for both healpix and rectangular pixels"""
    if not(healpix): 
        assert imap.ndim >= 2
        return cs.map2alm(imap,lmax=lmax)
    else:
        return hp.map2alm(imap,lmax=lmax,iter=iter)




def isotropic_filter_T(imap=None,alm=None,lcltt=None,ucltt=None,
                       nltt_deconvolved=None,tcltt=None,lmin=None,lmax=None,gradient=False):
    if imap is not None:
        imap -= imap.mean()
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



def qe_tt_simple(Xmap=None,Ymap=None,Xalm=None,Yalm=None,lcltt=None,ucltt=None, \
                 nltt_deconvolved=None,tcltt=None,nltt_deconvolved_y=None,tcltt_y=None,
                 lmin=None,lmax=None,lmin_y=None,lmax_y=None,do_curl=False,mlmax=None,healpix=False,pure_healpix=False,
                 shape=None,wcs=None):
    """
    Does -div(grad(wX)*wY) where wX and wY are Wiener filtered appropriately for isotropic noise
    from provided X and Y, and CMB and noise spectra.
    Does not normalize the estimator.
    """
    
    # Set defaults if the high-res map Y is not different from the gradient map X
    if Ymap is None: assert (nltt_deconvolved_y is None) and (tcltt_y is None) and (lmin_y is None) and (lmax_y is None)
    if nltt_deconvolved_y is None: nltt_deconvolved_y = nltt_deconvolved
    if tcltt_y is None: tcltt_y = tcltt
    if lmin_y is None: lmin_y = lmin
    if lmax_y is None: lmax_y = lmax

    # lmax at which harmonic operations are performed
    #if mlmax is None: mlmax = max(lmax,lmax_y) + 250
    #if mlmax is None: mlmax = 2*max(lmax,lmax_y)
    if mlmax is None: mlmax = max(lmax,lmax_y)+250
    if pure_healpix: mlmax = None

    # if healpix, then calculate intermediate CAR geometry
    if healpix:
        npix = Xmap.size
        shape,wcs = enmap.fullsky_geometry(res=get_fullsky_res(npix=npix),proj="car")
    else:
        if (shape is None) or (wcs is None): shape,wcs = Xmap.shape,Xmap.wcs

    # map -> alm
    if Xalm is None:
        Xmap -= Xmap.mean()
        Xalm = gmap2alm(Xmap,lmax=mlmax,healpix=healpix)
        del Xmap
    if Yalm is None:
        if Ymap is None:
            Yalm = Xalm.copy()
        else:
            Ymap -= Ymap.mean()
            Yalm = gmap2alm(Ymap,lmax=mlmax,healpix=healpix)
        del Ymap

    # filter alms
    iXalm = isotropic_filter_T(alm=Xalm,lcltt=lcltt,ucltt=ucltt,
                              nltt_deconvolved=nltt_deconvolved,tcltt=tcltt,lmin=lmin,lmax=lmax,gradient=True)
    iYalm = isotropic_filter_T(alm=Yalm,lcltt=lcltt,ucltt=ucltt,
                              nltt_deconvolved=nltt_deconvolved_y,tcltt=tcltt_y,lmin=lmin_y,lmax=lmax_y,gradient=False)
    # get kappa
    if pure_healpix:
        from falafel import qehp
        nside = hp.npix2nside(npix)
        return qehp.qe_tt(nside,iXalm,iYalm,mlmax=mlmax,do_curl=do_curl,lmax_x=lmax,lmax_y=lmax_y)
    else:    
        return qe_tt(shape,wcs,iXalm,iYalm,mlmax=mlmax,do_curl=do_curl,lmax_x=lmax,lmax_y=lmax_y)
    
def qe_tt(shape,wcs,Xalm,Yalm,do_curl=False,mlmax=None,lmax_x=None,lmax_y=None):
    """
    Does -div(grad(wX)*wY) where wX_alm and wY_alm are provided as appropriately Wiener filtered alms.
    Does not normalize the estimator.
    """

    # Filters to impose hard ell cuts on output alms
    if (lmax_x is not None) or (lmax_y is not None):
        ells = np.arange(mlmax)
        lxymax = max(lmax_x,lmax_y)
        xyfil = np.ones(mlmax)
        xyfil[ells>lxymax] = 0
    if lmax_x is not None:
        xfil = np.ones(mlmax)
        xfil[ells>lmax_x] = 0
        Xalm = hp.almxfl(Xalm,xfil)
    if lmax_y is not None:
        yfil = np.ones(mlmax)
        yfil[ells>lmax_y] = 0
        Yalm = hp.almxfl(Yalm,yfil)

    ells = np.arange(0,mlmax)
    fil = np.sqrt(ells*(ells+1.))
    fil[ells>lmax_y] = 0
    kalms = hp.almxfl(-qe_spin_tt(shape,wcs,Xalm,Yalm,lmax_x,mlmax)[0],fil)
    calms = hp.almxfl(-qe_spin_tt(shape,wcs,Xalm,Yalm,lmax_x,mlmax)[1],fil)
    return kalms, calms
        
