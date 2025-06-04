from pixell import curvedsky as cs, enmap
import numpy as np
import healpy as hp
import sys

"""

"""

fudge = True

class pixelization(object):
    def __init__(self,shape=None,wcs=None,nside=None,dtype=np.float32,iter=0):
        self.dtype = dtype
        self.ctype = {np.float32: np.complex64, np.float64: np.complex128}[dtype]
        if shape is not None:
            assert wcs is not None
            assert nside is None
            self.hpix = False
            self.shape, self.wcs = shape[-2:],wcs
        else:
            assert wcs is None
            assert nside is not None
            self.hpix = True
            self.nside = nside
            self.iter = iter


    def white_noise(self,noise_muK_arcmin,seed=None):
        """Generate a white noise map
        """
        pass


    def alm2map_spin(self,alm,spin_alm,spin_transform,ncomp,mlmax):
        """
        Returns
        X(n) = sum_lm  alm_s1 s2_Y_lm(n)

        where s1=spin_alm and s2=spin_transform are different spins.

        alm should be (alm_+|s|, alm_-|s|)

        """
        # Transform (alm_+|s|, alm_-|s|) to (alm_+, alm_-)
        ap_am = irot2d(alm,spin=spin_alm)
        if self.hpix:
            res = hp.alm2map_spin(ap_am,nside=self.nside,spin=abs(spin_transform),lmax=mlmax)
        else:
            omap = enmap.empty((ncomp,)+self.shape[-2:],self.wcs,dtype=self.dtype)
            res = cs.alm2map(ap_am,omap,spin=abs(spin_transform))
        # Rotate maps M+ and M- to M_+|s| and M_-|s|
        # M_+|s| is the first component
        # M_-|s| is the second component
        return rot2d(res)

    def alm2map(self,alm,spin,ncomp,mlmax):
        if self.hpix:
            if spin!=0: 
                res = hp.alm2map_spin(alm,nside=self.nside,spin=spin,lmax=mlmax)
                return res
            else: 
                # complex64 not supported here
                return hp.alm2map(alm.astype(np.complex128),nside=self.nside,pol=False)[None]
        else:
            omap = enmap.empty((ncomp,)+self.shape,self.wcs,dtype=self.dtype)
            return cs.alm2map(alm,omap,spin=spin)
        

    def map2alm(self,imap,lmax,tweak=True):
        if self.hpix:
            return hp.map2alm(imap,lmax=lmax,iter=self.iter)
        else:
            return cs.map2alm(imap,lmax=lmax,tweak=True)

    def map2alm_spin(self,imap,lmax,spin_alm,spin_transform):
        dmap = -irot2d(np.stack((imap,imap.conj())),spin=spin_alm).real
        if self.hpix:
            res = hp.map2alm_spin(dmap,lmax=lmax,spin=spin_transform)
            return res
        else:
            return cs.map2alm(enmap.enmap(dmap,imap.wcs),spin=spin_transform,lmax=lmax,tweak=True)


def get_mlmax(alms):
    if alms.ndim==2:
        asize = alms[0].size
    elif alms.ndim == 1:
        asize = alms.size
    else:
        print(alms.shape)
        raise ValueError
    return hp.Alm.getlmax(asize)

def filter_alms(alms,filt,lmin=None,lmax=None):
    """
    Filter the alms with transfer function specified
    by filt (indexed starting at ell=0).
    """
    mlmax = get_mlmax(alms)
    ls = np.arange(filt.size)
    if lmax is not None:
        assert lmax<=ls.max()
        assert lmax<=mlmax
    if lmin is not None: filt[ls<lmin] = 0
    if lmax is not None: filt[ls>lmax] = 0
    return cs.almxfl(alms.copy(),filt)

def rot2d(fmap):
    """
    If fmap is an [2,...] ndarray with f0=fmap[0],f1=fmap[1],
    returns [( f0 + i f1 ) , ( f0 - i f1)]
    e.g. Rotates the map outputs M+ and M- of alm2map into sM and -sM
    """
    return np.stack((fmap[0]+fmap[1]*1j,fmap[0]-fmap[1]*1j))

def irot2d(fmap,spin):
    """
    If fmap is an [2,...] ndarray with f0=fmap[0],f1=fmap[1],
    returns [( f0 + (-1)^s f1 )/2 , ( f0 - (-1)^s f1 )/(2i)]
    e.g. Rotates the alms +sAlm and -sAlm into inputs a+ and a- for map2alm
    """
    return -np.stack(((fmap[0]+((-1)**spin)*fmap[1])/2.,(fmap[0]-((-1)**spin)*fmap[1])/2./1j))

def rot2dalm(fmap,spin):
    #inverse operation of irot2d
    ps=(fmap[0]+1j*fmap[1])
    ms=(-1)**spin*(fmap[0]-1j*fmap[1])
    return -np.stack((ps,ms))


def almxfl(alm,fl):
    alm = np.asarray(alm)
    ncomp = alm.shape[0]
    assert ncomp in [1,2,3]
    res = cs.almxfl(alm,fl)
    return res

def pol_alms(Ealm,Balm): 
    return np.stack((Ealm+1j*Balm,Ealm-1j*Balm))

def gradient_spin(px,alm,mlmax,spin):
    """
    Given appropriately Wiener filtered temperature map alms,
    returns a real-space map containing the gradient.

    alm should be (alm_+|s|, alm_-|s|)

    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """

    ells = np.arange(0,mlmax)
    if spin==0:
        fl = np.sqrt(ells*(ells+1.))
        spin_out = 1 ; comp = 0
        sign = 1
    elif spin==(-2):
        fl = ells * 0
        fl[ells>=1] = np.sqrt((ells[ells>=1]-1)*(ells[ells>=1]+2.))
        spin_out = -1 ; comp = 1
        if fudge:
            sign = 1 #!!! this sign is not understood
        else:
            sign = -1
    elif spin==2:
        fl = ells * 0
        fl[ells>=2] = np.sqrt((ells[ells>=2]-2)*(ells[ells>=2]+3.))
        spin_out = 3 ; comp = 0
        sign = -1
    fl[ells<2] = 0
    salms = almxfl(alm,fl)
    return sign*px.alm2map_spin(salms,spin,spin_out,ncomp=2,mlmax=mlmax)[comp]

def deflection_map_to_phi_curl_alms(px,dmap,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """

    res = px.map2alm_spin(dmap,lmax=mlmax,spin_alm=0,spin_transform=1)
    ells = np.arange(0,mlmax)
    fl = np.sqrt(ells*(ells+1.))
    res = almxfl(res,fl)
    return res

def qe_spin_temperature_deflection(px,Xalm,Yalm,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """
    grad = gradient_spin(px,np.stack((Xalm,Xalm)),mlmax,spin=0)
    ymap = px.alm2map(Yalm,spin=0,ncomp=1,mlmax=mlmax)[0]
    prod = -grad*ymap
    return prod

def qe_spin_pol_deflection(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """

    palms = pol_alms(X_Ealm,X_Balm)
    # palms is now (Ealm + i Balm, Ealm - i Balm)
    # corresponding to +2 and -2 spin components
    grad_p2 = gradient_spin(px,palms,mlmax,spin=2)
    grad_m2 = gradient_spin(px,palms,mlmax,spin=-2)
    # E_alm, B_alm -> Q(n), U(n) -> Q+iU, Q-iU
    ymap = rot2d(px.alm2map(np.stack((Y_Ealm,Y_Balm)),spin=2,ncomp=2,mlmax=mlmax))
    prod = -grad_m2*ymap[0]-grad_p2*ymap[1]
    if not(px.hpix):
        prod = enmap.enmap(prod,px.wcs)
    if fudge:
        return prod/2 # !! this factor of 2 is not understood
    else:
        return prod
    
def qe_temperature_only(px,Xalm,Yalm,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """

    dmap = qe_spin_temperature_deflection(px,Xalm,Yalm,mlmax)
    return deflection_map_to_phi_curl_alms(px,dmap,mlmax)

def qe_pol_only(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """
    dmap = qe_spin_pol_deflection(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax)
    return deflection_map_to_phi_curl_alms(px,dmap,mlmax)

def qe_mv(px,X_Talm,X_Ealm,X_Balm,Y_Talm,Y_Ealm,Y_Balm,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """
    dmap_t = qe_spin_temperature_deflection(px,X_Talm,Y_Talm,mlmax)
    dmap_p = qe_spin_pol_deflection(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax)
    return deflection_map_to_phi_curl_alms(px,dmap_t+dmap_p,mlmax),dmap_t,dmap_p


def qe_all(px,response_cls_dict,mlmax,
           fTalm=None,fEalm=None,fBalm=None,
           estimators=['TT','TE','EE','EB','TB','mv','mvpol'],
           xfTalm=None,xfEalm=None,xfBalm=None):
    """
    Returns reconstructed unnormalized estimators.

    Inputs are Cinv filtered alms.
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """
    ests = estimators
    kfunc = lambda x: deflection_map_to_phi_curl_alms(px,x,mlmax)

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
        """wiener filter and combine together the alms in list_spec"""
        res = 0
        for spec,alm in zip(list_spec,list_alms):
            res = res + filter_alms(alm,response_cls_dict[spec])
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

    test = lambda x: qe_spin_temperature_deflection(px,x,fTalm,mlmax)
    pest = lambda u,v,w,x: qe_spin_pol_deflection(px,u,v,w,x,mlmax)
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
    if ('mvpol' in ests) or ('MVPOL' in ests): 
        r_mvpol = kfunc(dmap('Peb'))
    if ('mvpol' in ests): results['mvpol'] = r_mvpol
    if ('MVPOL' in ests): results['MVPOL'] = r_mvpol
    if ('MV' in ests) or ('mv' in ests):
        r_mv = kfunc(dmap('Pteb')+dmap('Tte'))
    if 'mv' in ests: results['mv'] = r_mv
    if 'MV' in ests: results['MV'] = r_mv
    return results

def qe_mask(px,response_cls_dict,mlmax,fTalm,xfTalm=None):
    """
    Inputs are Cinv filtered alms.
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    output: Mask estimator alms
    """
    if np.shape(fTalm)[0] > 1 and np.shape(fTalm)[0] < 4:
        fTalm = fTalm[0]

    if xfTalm is None:
        xfTalm = fTalm.copy()
    tw = filter_alms(fTalm,response_cls_dict['TT'])
    rmapT=np.real(px.alm2map_spin(np.stack((tw,tw)),0,0,ncomp=2,mlmax=mlmax))
    rmap=np.real(px.alm2map_spin(np.stack((fTalm,fTalm)),0,0,ncomp=2,mlmax=mlmax))
    #multiply the two fields together
    prodmap=rmap*rmapT
    if not(px.hpix): prodmap=enmap.enmap(prodmap,px.wcs)
    realsp=prodmap[0] #spin +0 real space  field
    res=px.map2alm(realsp,mlmax, tweak=True)
    return res

def qe_shear(px,mlmax,Talm=None,fTalm=None):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    output: curved sky shear estimator
    """
    ells = np.arange(mlmax)
    #prepare temperature map
    rmapT=px.alm2map(np.stack((Talm,Talm)),spin=0,ncomp=1,mlmax=mlmax)[0]
    #find tbarf
    t_alm=cs.almxfl(fTalm,np.sqrt((ells-1.)*ells*(ells+1.)*(ells+2.)))
    alms=np.stack((t_alm,t_alm))
    rmap=px.alm2map_spin(alms,0,2,ncomp=2,mlmax=mlmax)   #same as 2 2
    #multiply the two fields together
    prodmap=rmap*rmapT
    if not(px.hpix): prodmap=enmap.enmap(prodmap,px.wcs)
    realsp2=prodmap[0] #spin +2 real space real space field
    realsm2=prodmap[1] #spin -2 real space real space field
    if not(px.hpix): 
        realsp2 = enmap.enmap(realsp2,px.wcs)
        realsm2=enmap.enmap(realsm2,px.wcs)
    #convert the above spin2 fields to spin pm 2 alms
    res1 = px.map2alm_spin(realsp2,mlmax,2,2) #will return pm2 
    res2= px.map2alm_spin(realsm2,mlmax,-2,2) #will return pm2
    #spin 2 ylm 
    ttalmsp2=rot2dalm(res1,2)[0] #pick up the spin 2 alm of the first one
    ttalmsm2=rot2dalm(res1,2)[1] #pick up the spin -2 alm of the second one
    shear_alm=ttalmsp2+ttalmsm2
    return shear_alm

def qe_m4(px,mlmax,Talm=None,fTalm=None):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    output: curved sky multipole=4 estimator
    """
    ells = np.arange(mlmax)
    #prepare temperature map
    rmapT=px.alm2map(np.stack((Talm,Talm)),spin=0,ncomp=1,mlmax=mlmax)[0]
    #find tbarf
    t_alm=cs.almxfl(fTalm,np.sqrt((ells-3.)*(ells-2.)*(ells-1.)*ells*(ells+1.)*(ells+2.)*(ells+3.)*(ells+4.)))

    alms=np.stack((t_alm,t_alm))
    rmap=px.alm2map_spin(alms,0,4,ncomp=2,mlmax=mlmax)

    #multiply the two fields together
    rmap=np.nan_to_num(rmap)
    prodmap=rmap*rmapT
    prodmap=np.nan_to_num(prodmap)
    if not(px.hpix): prodmap=enmap.enmap(prodmap,px.wcs)
    realsp2=prodmap[0] #spin +4 real space real space field
    if not(px.hpix): realsp2 = enmap.enmap(realsp2,px.wcs)
    #convert the above spin4 fields to spin pm 4 alms
    res1 = px.map2alm_spin(realsp2,mlmax,4,4) #will return pm4
    #spin 4 ylm 
    ttalmsp2=rot2dalm(res1,4)[0] #pick up the spin 4 alm of the first one
    ttalmsm2=rot2dalm(res1,4)[1] #pick up the spin -4 alm of the second one
    m4_alm=ttalmsp2+ttalmsm2
    return m4_alm


def qe_source(px,mlmax,fTalm,profile=None,xfTalm=None):
    """generalised source estimator

    Args:
        px (object): pixelization object
        mlmax (int): maximum ell to perform alm2map transforms
        profile (narray): profile of reconstructed source in ell space. 
                          If none is provided, defaults to point source hardening
        fTalm (narray): inverse filtered temperature map
        xfTalm (narray, optional): inverse filtered temperature map. Defaults to None

    Returns:
        narray:  profile reconstruction
    """
    if profile is not None:
        #filter the first map with the source profile
        fTalm = cs.almxfl(fTalm, profile)
    #If we don't provide a second map,
    #copy the first (which is already filtered)
    if xfTalm is None:
        xfTalm = fTalm.copy()
    else:
        #otherwise, we still need to filter
        #the second map
        if profile is not None:
            xfTalm = cs.almxfl(xfTalm, profile)
    rmap1 = px.alm2map(fTalm,spin=0,ncomp=1,mlmax=mlmax)[0]
    rmap2 = px.alm2map(xfTalm,spin=0,ncomp=1,mlmax=mlmax)[0]
    #multiply the two fields together
    prodmap=rmap1*rmap2
    if not(px.hpix): prodmap=enmap.enmap(prodmap,px.wcs) #spin +0 real space  field
    res=px.map2alm_spin(prodmap,mlmax,0,0)
    #spin 0 salm 
    salm=0.5*res[0]
    if profile is not None:
        salm=cs.almxfl(salm,1./profile)
    return salm

def qe_rot(px,response_cls_dict,mlmax,fEalms,fBalms):
    """
    Inputs are Cinv filtered E and B alms.
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    fEalms, fBalms = Inverse Variance filtered E and B alms
    output: Birefringence angle alpha alms
    """
    fEalms = filter_alms(fEalms,response_cls_dict['EE'])
    zeros = np.zeros(fEalms.shape)
    # get Q and U comps from filtered E and B
    qE,uE = cs.alm2map(np.array([fEalms,zeros]),enmap.ndmap(np.zeros((2,)+px.shape),px.wcs),spin=[2],tweak=True)
    qB,uB = cs.alm2map(np.array([zeros,fBalms]),enmap.ndmap(np.zeros((2,)+px.shape),px.wcs),spin=[2],tweak=True)
    # make the biref map
    diffmap = (uE*qB)-(qE*uB)
    alpha_alm=-2*px.map2alm(diffmap,mlmax,tweak=True) #added tweak=Trues on Mar 31
    return alpha_alm

def qe_tau_pol(px,response_cls_dict,mlmax,fEalms,fBalms):
    """
    Inputs are Cinv filtered alms.
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    fEalms, fBalms = Inverse Variance filtered E and B alms
    output: POLARIZATION tau alms
    """
    # apply Weiner filter to E alms, to upweight the E dominant modes
    fEalms = filter_alms(fEalms,response_cls_dict['EE']) 
    zeros = np.zeros(fEalms.shape)
    # get Q and U comps from filtered E and B
    qE,uE = cs.alm2map(np.array([fEalms,zeros]),enmap.ndmap(np.zeros((2,)+px.shape),px.wcs),spin=[2])
    qB,uB = cs.alm2map(np.array([zeros,fBalms]),enmap.ndmap(np.zeros((2,)+px.shape),px.wcs),spin=[2])
    # make the tau map
    diffmap = (uE*uB)+(qE*qB)
    tau_alm = -px.map2alm(diffmap,mlmax)
    return tau_alm
