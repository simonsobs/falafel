from pixell import curvedsky as cs, enmap
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent

def filter_alms(alms,ffunc,lmin=None,lmax=None):
    mlmax = hp.Alm.getlmax(alms.size)
    ells = np.arange(0,mlmax)
    filt = np.nan_to_num(ffunc(ells))+ells*0
    if lmin is not None: filt[ells<lmin] = 0
    if lmax is not None: filt[ells>lmax] = 0
    return hp.almxfl(alms.copy(),filt)

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

    where s1=spin_alm and s2=spin_transform are different spins.

    alm should be (alm_+|s|, alm_-|s|)

    """
    # Transform (alm_+|s|, alm_-|s|) to (alm_+, alm_-)
    ap_am = irot2d(alm,spin=spin_alm)
    # Rotate maps M+ and M- to M_+|s| and M_-|s|
    res = rot2d(cs.alm2map(ap_am,omap,spin=abs(spin_transform)))
    # M_+|s| is the first component
    # M_-|s| is the second component
    return res

def almxfl(alm,fl):
    ncomp = alm.shape[0]
    assert ncomp in [1,2,3]
    res = alm.copy()
    for i in range(ncomp): res[i] = hp.almxfl(alm[i],fl)
    return res

def pol_alms(Ealm,Balm): return np.stack((Ealm+1j*Balm,Ealm-1j*Balm))

def gradient_spin(shape,wcs,alm,mlmax,spin):
    """
    Given appropriately Wiener filtered temperature map alms,
    returns a real-space map containing the gradient.

    alm should be (alm_+|s|, alm_-|s|)
    """
    omap = enmap.zeros((2,)+shape[-2:],wcs)
    ells = np.arange(0,mlmax)
    if spin==0:
        fl = np.sqrt(ells*(ells+1.))
        spin_out = 1 ; comp = 0
        sign = 1
    elif spin==(-2):
        fl = np.sqrt((ells-1)*(ells+2.))
        spin_out = -1 ; comp = 1
        sign = 1 #!!! this sign is not understood
    elif spin==2:
        fl = np.sqrt((ells-2)*(ells+3.))
        spin_out = 3 ; comp = 0
        sign = -1
    fl[ells<2] = 0
    salms = almxfl(alm,fl)
    return sign*alm2map_spin(salms,spin,spin_out,omap)[comp]

def deflection_map_to_kappa_curl_alms(dmap,mlmax):
    res = cs.map2alm(enmap.enmap(-irot2d(np.stack((dmap,dmap.conj())),spin=0).real,dmap.wcs),spin=1,lmax=mlmax)
    ells = np.arange(0,mlmax)
    fl = np.sqrt(ells*(ells+1.))/2
    res = almxfl(res,fl)
    return res

def qe_spin_temperature_deflection(shape,wcs,Xalm,Yalm,mlmax):
    grad = gradient_spin(shape,wcs,np.stack((Xalm,Xalm)),mlmax,spin=0)
    ymap = cs.alm2map(Yalm,enmap.zeros(shape[-2:],wcs))
    prod = -grad*ymap
    return enmap.enmap(prod,wcs)

def qe_spin_pol_deflection(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax):
    palms = pol_alms(X_Ealm,X_Balm)
    # palms is now (Ealm + i Balm, Ealm - i Balm)
    # corresponding to +2 and -2 spin components
    grad_p2 = gradient_spin(shape,wcs,palms,mlmax,spin=2)
    grad_m2 = gradient_spin(shape,wcs,palms,mlmax,spin=-2)
    # E_alm, B_alm -> Q(n), U(n) -> Q+iU, Q-iU
    ymap = rot2d(cs.alm2map(np.stack((Y_Ealm,Y_Balm)),enmap.zeros((2,)+shape[-2:],wcs),spin=2))
    prod = -grad_m2*ymap[0]-grad_p2*ymap[1]
    return enmap.enmap(prod,wcs)/2 # !! this factor of 2 is not understood
    
def qe_temperature_only(shape,wcs,Xalm,Yalm,mlmax):
    dmap = qe_spin_temperature_deflection(shape,wcs,Xalm,Yalm,mlmax)
    return deflection_map_to_kappa_curl_alms(dmap,mlmax)

def qe_pol_only(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax):
    dmap = qe_spin_pol_deflection(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax)
    return deflection_map_to_kappa_curl_alms(dmap,mlmax)

def qe_mv(shape,wcs,X_Talm,X_Ealm,X_Balm,Y_Talm,Y_Ealm,Y_Balm,mlmax):
    dmap_t = qe_spin_temperature_deflection(shape,wcs,X_Talm,Y_Talm,mlmax)
    dmap_p = qe_spin_pol_deflection(shape,wcs,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax)
    return deflection_map_to_kappa_curl_alms(dmap_t+dmap_p,mlmax),dmap_t,dmap_p


def qe_all(shape,wcs,theory_func,mlmax,fTalm=None,fEalm=None,fBalm=None,estimators=['TT','TE','EE','EB','TB','mv','mvpol'],xfTalm=None,xfEalm=None,xfBalm=None):
    """
    Inputs are Cinv filtered alms.
    """
    ests = estimators
    ells = np.arange(mlmax)
    th = lambda x,y: theory_func(x,y)
    kfunc = lambda x: deflection_map_to_kappa_curl_alms(x,mlmax)

    if xfTalm is None:
        if fTalm is not None: xfTalm = fTalm.copy()
    if xfEalm is None:
        if fEalm is not None: xfEalm = fEalm.copy()
    if xfBalm is None:
        if fBalm is not None: xfBalm = fBalm.copy()

    results = {}
    dcache = {}
    acache = {}

    def mixing(list_spec,list_alms):
        res = 0
        for spec,alm in zip(list_spec,list_alms): res = res + filter_alms(alm,lambda x: th(spec,x))
        return res

    def xalm(name):
        try: return acache[name]
        except:
            if name=='t':
                acache[name] = mixing(['TT','TE'],[xfTalm,xfEalm])
            elif name=='t_e0':
                acache[name] = mixing(['TE'],[xfEalm])
            elif name=='t0':
                acache[name] = mixing(['TT'],[xfTalm])
            elif name=='e':
                acache[name] = mixing(['EE','TE'],[xfEalm,xfTalm])
            elif name=='e_t0':
                acache[name] = mixing(['TE'],[xfTalm])
            elif name=='e0':
                acache[name] = mixing(['EE'],[xfEalm])
            elif name=='b':
                acache[name] = mixing(['BB'],[xfBalm])
            return acache[name]

    test = lambda x: qe_spin_temperature_deflection(shape,wcs,x,fTalm,mlmax)
    pest = lambda u,v,w,x: qe_spin_pol_deflection(shape,wcs,u,v,w,x,mlmax)
    try: zero = fTalm*0
    except: zero = fEalm*0

    def dmap(name):
        try: return dcache[name]
        except:
            if name=='Tte' : dcache[name] = test(xalm("t"))
            if name=='Tte0' : dcache[name] = test(xalm("t_e0"))
            if name=='Tt'  : dcache[name] = test(xalm("t0"))
            if name=='Pte' : dcache[name] = pest(xalm("e"),zero,fEalm,zero)
            if name=='Pe'  : dcache[name] = pest(xalm("e0"),zero,fEalm,zero)
            if name=='Peb' : dcache[name] = pest(xalm("e0"),zero,fEalm,fBalm) # note xalm("b") always set to zero
            if name=='Ptb' : dcache[name] = pest(xalm("e_t0"),zero,zero,fBalm) # otherwise TB ends up wrong
            if name=='Pte0': dcache[name] = pest(xalm("e_t0"),zero,fEalm,zero) # Another weird one for TE
            if name=='Peb0': dcache[name] = pest(xalm("e0"),zero,zero,fBalm) # Another weird one for EB
            if name=='Pteb': dcache[name] = pest(xalm("e"),zero,fEalm,fBalm)
            return dcache[name]

    """
    Functions:
    test(gT,gE | E)
    polest(gT,gE | E,B)


    TT: test(gT,gE=0 | T)
    EE: polest(gT=0,gE | E,B=0)
    EB: polest(gT=0,gE | E=0,B)
    TE: polest(gE=0,gT | E,B=0) + test(gT=0,gE | T)
    TB: polest(gT,gE=0 | E=0,B) # this one's a bit weird
    mvpol: polest(gT=0,gE | E,B)
    mv: polest(gT,gE | E,B) + test(gT,gE | T)

    """

    if 'TE' in ests: results['TE'] = kfunc(dmap('Tte0') + dmap('Pte0'))
    if 'EB' in ests: results['EB'] = kfunc(dmap('Peb0'))
    if 'EE' in ests: results['EE'] = kfunc(dmap('Pe'))
    if 'TT' in ests: results['TT'] = kfunc(dmap('Tt'))
    if 'TB' in ests: results['TB'] = kfunc(dmap('Ptb'))
    if 'mvpol' in ests: results['mvpol'] = kfunc(dmap('Peb'))
    if 'mv' in ests: results['mv'] = kfunc(dmap('Pteb')+dmap('Tte'))
    return results



def symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=100,lmax=2000,plot=True):
    import symlens
    from orphics import maps,stats,io
    shape,wcs = maps.rect_geometry(width_deg=80.,px_res_arcmin=2.0*3000./lmax)
    emin = maps.minimum_ell(shape,wcs)
    modlmap = enmap.modlmap(shape,wcs)
    tctt = maps.interp(range(len(tctt)),tctt)(modlmap)
    uctt = maps.interp(range(len(uctt)),uctt)(modlmap)
    tcee = maps.interp(range(len(tcee)),tcee)(modlmap)
    ucee = maps.interp(range(len(ucee)),ucee)(modlmap)
    tcbb = maps.interp(range(len(tcbb)),tcbb)(modlmap)
    ucbb = maps.interp(range(len(ucbb)),ucbb)(modlmap)
    ucte = maps.interp(range(len(ucte)),ucte)(modlmap)
    tcte = maps.interp(range(len(tcte)),tcte)(modlmap)
    
    tmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    feed_dict = {
        'uC_T_T':uctt,
        'tC_T_T':tctt,
        'uC_E_E':ucee,
        'tC_E_E':tcee,
        'uC_T_E':ucte,
        'tC_T_E':tcte,
        'tC_B_B':tcbb,
        'uC_B_B':ucbb,
    }
    polcombs = ['TT','TE','EE','TB','EB']
    Als = {}
    Al1ds = {}
    bin_edges = np.arange(3*emin,lmax,2*emin)
    binner = stats.bin2D(modlmap,bin_edges)
    alinv_mv = 0.
    alinv_mv_pol = 0.
    for pol in polcombs:
        Al = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator="hu_ok", XY=pol, xmask=tmask, ymask=tmask)
        cents,Al1d = binner.bin(Al)
        ls = np.arange(0,cents.max(),1)
        Als[pol] = np.interp(ls,cents,Al1d*cents**2.)/ls**2.
        Als[pol][ls<1] = 0
        Al1ds[pol] = Al1d.copy()
        alinv_mv += (1./Als[pol])
        if pol=='EE' or pol=='EB': alinv_mv_pol += (1./Als[pol])
    al_mv = (1./alinv_mv)
    al_mv[ls<1] = 0
    al_mv_pol = (1./alinv_mv_pol)
    al_mv_pol[ls<1] = 0
    if plot:
        pl = io.Plotter(xyscale='loglog',xlabel='',ylabel='')
        pl.add(ells,clkk,color='k',lw=3)
        pl.add(ls,al_mv*ls**2.,ls="-",color="red",label='mv',lw=2)
        pl.add(ls,al_mv_pol*ls**2.,ls="-",color="green",label='mv_pol',lw=2)
        for i,pol in enumerate(polcombs):
            pl.add(cents,Al1ds[pol]*cents**2.,color="C%d" % i,label=pol)
            pl.add(ls,Als[pol]*ls**2.,ls="--",color="C%d" % i)
        pl.done()


    Al1 = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator="hdv", XY="TE", xmask=tmask, ymask=tmask)
    Al2 = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator="hdv", XY="ET", xmask=tmask, ymask=tmask)
    Al_te_hdv = 1./((1./Al1)+(1./Al2))
    cents,Al1d = binner.bin(Al_te_hdv)
    Al_te_hdv = np.interp(ls,cents,Al1d*cents**2.)/ls**2.
    Al_te_hdv[ls<1] = 0
        
    return ls,Als,al_mv_pol,al_mv,Al_te_hdv
