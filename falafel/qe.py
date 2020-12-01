from pixell import curvedsky as cs, enmap
import numpy as np
import healpy as hp # needed only for isotropic filtering and alm -> cl, need to make it healpy independent

"""

"""

fudge = True

class pixelization(object):
    def __init__(self,shape=None,wcs=None,nside=None,dtype=np.float64,iter=0):
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
            else: return hp.alm2map(alm,nside=self.nside,verbose=False,pol=False)[None]
        else:
            omap = enmap.empty((ncomp,)+self.shape,self.wcs,dtype=self.dtype)
            return cs.alm2map(alm,omap,spin=spin)
        

    def map2alm(self,imap,lmax):
        if self.hpix:
            return hp.map2alm(imap,lmax=lmax,iter=self.iter)
        else:
            return cs.map2alm(imap,lmax=lmax)

    def map2alm_spin(self,imap,lmax,spin_alm,spin_transform):
        dmap = -irot2d(np.stack((imap,imap.conj())),spin=spin_alm).real
        if self.hpix:
            res = hp.map2alm_spin(dmap,lmax=lmax,spin=spin_transform)
            return res
        else:
            return cs.map2alm(enmap.enmap(dmap,imap.wcs),spin=spin_transform,lmax=lmax)


def filter_alms(alms,ffunc,lmin=None,lmax=None):
    mlmax = hp.Alm.getlmax(alms.size)
    ells = np.arange(0,mlmax)
    filt = np.nan_to_num(ffunc(ells))+ells*0
    if lmin is not None: filt[ells<lmin] = 0
    if lmax is not None: filt[ells>lmax] = 0
    return hp.almxfl(alms.copy(),filt)

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
    res = alm.copy()
    for i in range(ncomp): res[i] = hp.almxfl(alm[i],fl)
    return res

def pol_alms(Ealm,Balm): return np.stack((Ealm+1j*Balm,Ealm-1j*Balm))

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
        fl = np.sqrt((ells-1)*(ells+2.))
        spin_out = -1 ; comp = 1
        if fudge:
            sign = 1 #!!! this sign is not understood
        else:
            sign = -1
    elif spin==2:
        fl = np.sqrt((ells-2)*(ells+3.))
        spin_out = 3 ; comp = 0
        sign = -1
    fl[ells<2] = 0
    salms = almxfl(alm,fl)
    return sign*px.alm2map_spin(salms,spin,spin_out,ncomp=2,mlmax=mlmax)[comp]

def deflection_map_to_kappa_curl_alms(px,dmap,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """

    res = px.map2alm_spin(dmap,lmax=mlmax,spin_alm=0,spin_transform=1)
    ells = np.arange(0,mlmax)
    fl = np.sqrt(ells*(ells+1.))/2
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
    return deflection_map_to_kappa_curl_alms(px,dmap,mlmax)

def qe_pol_only(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """
    dmap = qe_spin_pol_deflection(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax)
    return deflection_map_to_kappa_curl_alms(px,dmap,mlmax)

def qe_mv(px,X_Talm,X_Ealm,X_Balm,Y_Talm,Y_Ealm,Y_Balm,mlmax):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """
    dmap_t = qe_spin_temperature_deflection(px,X_Talm,Y_Talm,mlmax)
    dmap_p = qe_spin_pol_deflection(px,X_Ealm,X_Balm,Y_Ealm,Y_Balm,mlmax)
    return deflection_map_to_kappa_curl_alms(px,dmap_t+dmap_p,mlmax),dmap_t,dmap_p


def qe_all(px,theory_func,theory_crossfunc,mlmax,fTalm=None,fEalm=None,fBalm=None,estimators=['TT','TE','EE','EB','TB','mv','mvpol'],xfTalm=None,xfEalm=None,xfBalm=None):
    """
    Inputs are Cinv filtered alms.
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    """
    ests = estimators
    ells = np.arange(mlmax)
    th = lambda x,y: theory_func(x,y)
    th_cross=lambda x,y: theory_crossfunc(x,y)
    kfunc = lambda x: deflection_map_to_kappa_curl_alms(px,x,mlmax)

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
        for spec,alm in zip(list_spec,list_alms):
            if spec=='TT':
                res = res + filter_alms(alm,lambda x: th_cross(spec,x))
            else:
                res = res + filter_alms(alm,lambda x: th(spec,x))
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
    if 'mvpol' in ests: results['mvpol'] = kfunc(dmap('Peb'))
    if 'mv' in ests: results['mv'] = kfunc(dmap('Pteb')+dmap('Tte'))

    return results

def qe_mask(px,theory_func,theory_crossfunc,mlmax,fTalm=None,fEalm=None,fBalm=None,estimators=['TT','TE','EE','EB','TB','mv','mvpol'],xfTalm=None,xfEalm=None,xfBalm=None):
    """
    Inputs are Cinv filtered alms.
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    output: Mask estimator alms
    """
    ests = estimators
    ells = np.arange(mlmax)
    th = lambda x,y: theory_func(x,y)
    th_cross=lambda x,y: theory_crossfunc(x,y)
    kfunc = lambda x: deflection_map_to_kappa_curl_alms(px,x,mlmax)
    omap = enmap.zeros((2,)+px.shape,px.wcs) #load empty map with SO map wcs and shape

    if xfTalm is None:
        if fTalm is not None: xfTalm = fTalm.copy()
    if xfEalm is None:
        if fEalm is not None: xfEalm = fEalm.copy()
    if xfBalm is None:
        if fBalm is not None: xfBalm = fBalm.copy()
    filt = np.nan_to_num(th_cross('TT',ells))+ells*0
    tw=hp.almxfl(fTalm.copy(),filt)
    rmapT=px.alm2map_spin(np.stack((tw,tw)),0,0,ncomp=2,mlmax=mlmax)
    rmap=px.alm2map_spin(np.stack((fTalm,fTalm)),0,0,ncomp=2,mlmax=mlmax)
    #multiply the two fields together
    prodmap=rmap*rmapT
    prodmap=enmap.samewcs(prodmap,omap)
    realsp=prodmap[0] #spin +0 real space  field
    


    res=px.map2alm_spin(realsp,mlmax,0,0)

    #spin 0 alm 
    ttalmsp2=res[0] 
    
    return ttalmsp2

def rot2dalm(fmap,spin):
    #inverse operation of irot2d
    ps=(fmap[0]+1j*fmap[1])
    ms=(-1)**spin*(fmap[0]-1j*fmap[1])
    return -np.stack((ps,ms))


def qe_shear(px,mlmax,Talm=None,fTalm=None):
    """
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    output: curved sky shear estimator
    """
    ells = np.arange(mlmax)
    omap = enmap.zeros((2,)+px.shape,px.wcs) #load empty map with SO map wcs and shape
    #prepare temperature map
    rmapT=px.alm2map(np.stack((Talm,Talm)),spin=0,ncomp=1,mlmax=mlmax)[0]


    #find tbarf
    t_alm=hp.almxfl(fTalm,np.sqrt((ells-1.)*ells*(ells+1.)*(ells+2.)))
    alms=np.stack((t_alm,t_alm))
    rmap=px.alm2map_spin(alms,0,2,ncomp=2,mlmax=mlmax)   #same as 2 2
    #multiply the two fields together
    prodmap=rmap*rmapT
    prodmap=enmap.samewcs(prodmap,omap)
    realsp2=prodmap[0] #spin +2 real space real space field
    realsm2=prodmap[1] #spin -2 real space real space field
    realsp2 = enmap.samewcs(realsp2,omap)
    realsm2=enmap.samewcs(realsm2,omap)


    #convert the above spin2 fields to spin pm 2 alms
    res1 = px.map2alm_spin(realsp2,mlmax,2,2) #will return pm2 
    res2= px.map2alm_spin(realsm2,mlmax,-2,2) #will return pm2

    #spin 2 ylm 
    ttalmsp2=rot2dalm(res1,2)[0] #pick up the spin 2 alm of the first one
    ttalmsm2=rot2dalm(res1,2)[1] #pick up the spin -2 alm of the second one
    shear_alm=ttalmsp2+ttalmsm2

    
    
    return shear_alm

def qe_pointsources(px,theory_func,theory_crossfunc,mlmax,fTalm=None,fEalm=None,fBalm=None,estimators=['TT','TE','EE','EB','TB','mv','mvpol'],xfTalm=None,xfEalm=None,xfBalm=None):
    """
    Inputs are Cinv filtered alms.
    px is a pixelization object, initialized like this:
    px = pixelization(shape=shape,wcs=wcs) # for CAR
    px = pixelization(nside=nside) # for healpix
    output: Point source estimator alms
    """
    ests = estimators
    ells = np.arange(mlmax)
    th = lambda x,y: theory_func(x,y)
    th_cross=lambda x,y: theory_crossfunc(x,y)
    kfunc = lambda x: deflection_map_to_kappa_curl_alms(px,x,mlmax)
    omap = enmap.zeros((2,)+px.shape,px.wcs) #load empty map with SO map wcs and shape

    if xfTalm is None:
        if fTalm is not None: xfTalm = fTalm.copy()
    if xfEalm is None:
        if fEalm is not None: xfEalm = fEalm.copy()
    if xfBalm is None:
        if fBalm is not None: xfBalm = fBalm.copy()
    rmap=px.alm2map_spin(np.stack((fTalm,fTalm)),0,0,ncomp=2,mlmax=mlmax)
    #multiply the two fields together
    prodmap=rmap**2
    prodmap=enmap.samewcs(prodmap,omap)
    realsp=prodmap[0] #spin +0 real space  field
    


    res=px.map2alm_spin(realsp,mlmax,0,0)

    #spin 0 salm 
    salm=0.5*res[0] 
    
    return salm

def symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=100,lmax=2000,plot=True,estimator="hu_ok"):
    import symlens
    from orphics import maps,stats,io
    shape,wcs = maps.rect_geometry(width_deg=80.,px_res_arcmin=2.0*3000./lmax)
    emin = maps.minimum_ell(shape,wcs)
    modlmap = enmap.modlmap(shape,wcs)
    tctt = maps.interp(range(len(tctt)),tctt)(modlmap)
    ells=np.arange(len(uctt))
    ductt=np.gradient(np.log(uctt),np.log(ells))
    uctt = maps.interp(range(len(uctt)),uctt)(modlmap)
    ductt=maps.interp(range(len(ductt)),ductt)(modlmap)
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
        if estimator=="hu_ok" or estimator=="hdv":
            Al = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator=estimator, XY=pol, xmask=tmask, ymask=tmask)
        elif estimator=="shear":
            print("calculating shear norm and noise")
            ells=np.arange(len(uctt))
            feed_dict['duC_T_T'] =ductt
            Al = symlens.A_l(shape, wcs, feed_dict=feed_dict, estimator=estimator, XY="TT", xmask=tmask, ymask=tmask)
            Noise=symlens.qe.N_l(shape, wcs, feed_dict=feed_dict, estimator=estimator, XY="TT", xmask=tmask, ymask=tmask,Al=Al,field_names=None, kmask=None)
            cents,Ns1d = binner.bin(Noise)
            ls = np.arange(0,cents.max(),1)
            Ns=np.interp(ls,cents,Ns1d*cents**2.)/ls**2.
            Ns[ls<1] = 0
  
            cents,Al1d = binner.bin(Al)
            Als= np.interp(ls,cents,Al1d*cents**2.)/ls**2.
            Als[ls<1] = 0

            return ls,Als,Ns
            
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
