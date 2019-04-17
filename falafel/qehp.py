import healpy as hp
import numpy as np

def gradient_T_map(nside,alm):
    """
    Given appropriately Wiener filtered temperature map alms,
    returns a real-space map containing the gradient of T.
    """
    return hp.alm2map_der1(alm,nside)[1:]


def qe_tt(nside,Xalm,Yalm,do_curl=False,mlmax=None,lmax_x=None,lmax_y=None):
    """
    Does -div(grad(wX)*wY) where wX_alm and wY_alm are provided as appropriately Wiener filtered alms.
    Does not normalize the estimator.
    """

    # Filters to impose hard ell cuts on output alms
    flmax = mlmax if mlmax is not None else max(lmax_x,lmax_y)
    if (lmax_x is not None) or (lmax_y is not None):
        ells = np.arange(flmax)
        lxymax = max(lmax_x,lmax_y)
        xyfil = np.ones(flmax)
        xyfil[ells>lxymax] = 0
    if lmax_x is not None:
        xfil = np.ones(flmax)
        xfil[ells>lmax_x] = 0
        Xalm = hp.almxfl(Xalm,xfil)
    if lmax_y is not None:
        yfil = np.ones(flmax)
        yfil[ells>lmax_y] = 0
        Yalm = hp.almxfl(Yalm,yfil)
        
    
    # Get gradient and high-pass map in real space
    gradT0,gradT1 = gradient_T_map(nside,Xalm)
    highT = hp.alm2map(Yalm,nside)

    # Form real-space products of gradient and high-pass
    px = gradT0 * highT
    py = gradT1 * highT

    del highT
    
    # alms of products for divergence
    alm_px = hp.map2alm(px,lmax=mlmax)
    alm_py = hp.map2alm(py,lmax=mlmax)
    if (lmax_x is not None) or (lmax_y is not None):
        alm_px = hp.almxfl(alm_px,xyfil)
        alm_py = hp.almxfl(alm_py,xyfil)
    
    del px
    del py

    # divergence from alms
    dpxdx,dpxdy = hp.alm2map_der1(alm_px,nside)[1:]
    del alm_px
    dpydx,dpydy = hp.alm2map_der1(alm_py,nside)[1:]
    del alm_py

    # unnormalized kappa from divergence
    kappa = dpxdx + dpydy
    alm_kappa = -hp.map2alm(kappa,lmax=mlmax)
    if (lmax_x is not None) or (lmax_y is not None):
        alm_kappa = hp.almxfl(alm_kappa,xyfil)
    del kappa

    if do_curl:
        curl = dpydx - dpxdy
        del dpxdx,dpydy,dpydx,dpxdy
        alm_curl = -hp.map2alm(curl,lmax=mlmax)
        if (lmax_x is not None) or (lmax_y is not None):
            alm_curl = hp.almxfl(alm_curl,xyfil)
        del curl
        return alm_kappa,alm_curl
    else:
        del dpxdx,dpydy,dpydx,dpxdy
        return alm_kappa
    



