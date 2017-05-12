# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:25:47 2017

@author: quentinpeter
"""
import numpy as np
from .basis_generate import getprofiles
import scipy
gfilter=scipy.ndimage.filters.gaussian_filter1d
import warnings

def size_profiles(profiles,Q,Wz,pixsize,readingpos=None,Rs=None,*,
                  initmode='none',normalize_profiles=True,Zgrid=11,
                  ignore=10e-6,data_dict=None,fit_position_number=None,
                  central_profile=False):
    """Size the profiles
    
     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    flowRate: float
        Speed of the flow in [ul/h]
    Wz: float
        The channel height in [m]
    pixsize:float
        The pixel size in [m]
    readingpos: 1d float array
        The reading position of the profiles
    Rs: 1d float
        The test radii [m] 
    initmode: string, default 'none'
        How should the initial profile be processed 
        See init_process
    normalize_profiles: Bool, default True
        Should normalize profiles?
    Zgrid: integer, default 11
        Number of Z slices
    ignore: float, default 10e-6
        Ignore on the sides [m]
    data_dict: dictionnary
        If not None, returns infos
    fit_position_number: 1d array
        Posiitons to fit
    central_profile: Bool, default False
        Should use central profile?
      
    Returns
    -------
    radii: float
        The best radius fit
    """
    
    profiles=np.asarray(profiles)
    #normalize if needed
    if normalize_profiles:
        #if profile is mainly negative, error
        if np.any(np.sum(profiles*(profiles>0),1)<5*-np.sum(profiles*(profiles<0),1)):
            #raise RuntimeError("Profiles are negative!")
            return np.nan
        profiles/=np.sum(profiles,-1)[:,np.newaxis]
            
    """
    from matplotlib.pyplot import figure, plot
    figure()
    plot(np.ravel(profiles))
    #"""
    
    if fit_position_number is None:
        fit_position_number=np.arange(len(readingpos))
    else:
        fit_position_number=np.sort(fit_position_number)
        
    #treat init profile
    init=init_process(profiles[fit_position_number[0]],initmode)
        
    #Get best fit
    r=fit_monodisperse_radius([init,*profiles[fit_position_number[1:]]],
                                 readingpos=readingpos[fit_position_number],
                                 flowRate=Q,
                                 Wz=Wz,
                                 Zgrid=Zgrid,
                                 ignore=ignore,
                                 pixs=pixsize,
                                 Rs=Rs,
                                 central_profile=central_profile)
    
    if not r>0:
        return np.nan
    #fill data if needed
    if data_dict is not None:
        data_dict['initprof']=init
        if fit_position_number[0] !=0:
            init=init_process(profiles[0],initmode)
        data_dict['fits']=getprofiles(init,Q=Q, Radii=[r], 
                             Wy = len(init)*pixsize, Wz= Wz, Zgrid=Zgrid,
                             readingpos=readingpos[1:]-readingpos[0],
                                 central_profile=central_profile)[0]
        
    return r

def fit_monodisperse_radius(profiles, flowRate, pixs, readingpos,
                                Wz=50e-6,
                                Zgrid=11,
                                ignore=10e-6,
                                Rs=np.arange(.5,10,.5)*1e-9,
                                central_profile=False):
    """Find the best monodisperse radius
    
     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    flowRate: float
        Speed of the flow in [ul/h]
    pixs:float
        The pixel size in [m]
    readingpos: 1d float array
        The reading position of the profiles
    Wz: float, default 50e-6
        The channel height in [m]
    Zgrid: integer, default 11
        Number of Z slices
    ignore: float, default 10e-6
        Ignore on the sides [m]
    Rs: 1d float, default np.arange(.5,10,.5)*1e-6
        The test radii [m] 
    central_profile: Bool, default False
        If true, use the central profile
    
    Returns
    -------
    radii: float
        The best radius fit
    """
    profiles=np.asarray(profiles)
    #First reading pos is initial profile
    readingpos=readingpos[1:]-readingpos[0]
    #How many pixels should we ignore?
    ignore=int(ignore/pixs)
    if ignore ==0:
        ignore=1
    
    #Get basis function    
    Wy=pixs*np.shape(profiles)[1]
    
    
    
    Basis=getprofiles(profiles[0],flowRate,Rs,Wy=Wy,Wz=Wz,
                      Zgrid=Zgrid,readingpos=readingpos,
                      central_profile=central_profile)            
    
    #Compute residues
    p=profiles[1:]
    res=np.empty(len(Rs),dtype=float)
    for i,b in enumerate(Basis):
        res[i]=np.sqrt(np.mean(np.square(b-p)[:,ignore:-ignore]))

    '''
    from matplotlib.pyplot import figure, plot
    figure()
    plot(Rs,res)
    figure()
    plot(np.ravel(profiles[1:]))
    plot(np.ravel(Basis[np.argmin(res)]))
    #'''
    
    #Use linear combination between the two smallest results
    i,j=np.argsort(res)[:2]
    b1=Basis[i,:,ignore:-ignore]
    b2=Basis[j,:,ignore:-ignore]
    p0=p[:,ignore:-ignore]
    c=-np.sum((b1-b2)*(b2-p0))/np.sum((b1-b2)**2)
    
    #Get resulting r
    r=c*(Rs[i]-Rs[j])+Rs[j]
    return r

def center(prof):
    """
    Uses correlation between Y and the mirror image of Y to get the center
    
    Parameters
    ----------
    prof:  1d array
        Profile 
        
    Returns
    -------
    center: float
        The center position in pixel units
    
    """
    
    #We must now detect the position of the center. We use correlation
    #Correlation is equivalent to least squares (A-B)^2=-2AB+ some constants
    prof=np.array(prof)
    prof[np.isnan(prof)]=0
    Yi=prof[::-1]
    corr=np.correlate(prof,Yi, mode='full')
    X=np.arange(len(corr))
    args=np.argsort(corr)
    x=X[args[-7:]]
    y=corr[args[-7:]]
    coeffs=np.polyfit(x,np.log(y),2)
    center=-coeffs[1]/(2*coeffs[0])
    center=(center-(len(corr)-1)/2)/2+(len(prof)-1)/2
    return center

def baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile
    
    Parameters
    ----------
    prof:  1d array
        Profile 
    frac: float, defaults .05
        Fraction of the profile to use
        
    Returns
    -------
    baseline: 1d array
        The baseline
    
    """
    #we use 5% on left side to get the correct 0:
    #Get left and right zeros
    argvalid=np.argwhere(np.isfinite(prof))
    lims=np.squeeze([argvalid[0],argvalid[-1]])
    left=int(lims[0]+frac*np.diff(lims))
    right=int(lims[1]-frac*np.diff(lims))
    leftZero=np.nanmean(prof[lims[0]:left])
    rightZero=np.nanmean(prof[right:lims[1]])
        
    #Send profile to 0
    baseline=np.linspace(leftZero,rightZero,len(prof))
    return baseline

def flat_baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile
    
    Parameters
    ----------
    prof:  1d array
        Profile 
    frac: float, defaults .05
        Fraction of the profile to use
        
    Returns
    -------
    baseline: 1d array
        The flat baseline
    
    """
    #we use 5% on left side to get the correct 0:
    #Get left and right zeros
    leftZero=np.nanmean(prof[:int(frac*len(prof))])
    rightZero=np.nanmean(prof[-int(frac*len(prof)):])
        
    #Send profile to 0
    ret=np.zeros(prof.shape)+np.mean([leftZero,rightZero])
    return ret

def image_angle(image, maxAngle=np.pi/7):
    """
    Analyse an image with y invariance to extract a small angle.
    
    Parameters
    ----------
    image:  2d array
        image with y invariance 
    maxAngle: float, defaults np.pi/7
        Maximal rotation angle 
        
    Returns
    -------
    angle: float
        The rotation angle
    
    """
    #Difference left 50% with right 50%
    #We want to slice in two where we have data
    argvalid=np.argwhere(np.isfinite(np.nanmean(image,1)))
    lims=np.squeeze([argvalid[0],argvalid[-1]])
    #should we flatten this?
    top=np.nanmean(image[lims[0]:np.mean(lims,dtype=int)] ,0)
    bottom=np.nanmean(image[np.mean(lims,dtype=int):lims[1]],0)
    #Remouve nans
    top[np.isnan(top)]=0
    bottom[np.isnan(bottom)]=0
    #correlate
    C=np.correlate(top,bottom, mode='full')
    
    pos=np.arange(len(C))-(len(C)-1)/2
    disty=((lims[1]-lims[0])/2)
    Angles=np.arctan(pos/disty)
    
    valid=np.abs(Angles)<maxAngle
    x=pos[valid]
    c=C[valid]  

    x=x[c.argmax()-5:c.argmax()+6]
    y=np.log(gfilter(c,2)[c.argmax()-5:c.argmax()+6])  
    
    assert not np.any(np.isnan(y)), 'The signal is too noisy! '
        
    coeff=np.polyfit(x,y,2)
    x=-coeff[1]/(2*coeff[0])
    angle=np.arctan(x/disty)     
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arctan(pos/disty),C)
    plt.plot([maxAngle,maxAngle],[np.min(C),np.max(C)])
    plt.plot([-maxAngle,-maxAngle],[np.min(C),np.max(C)])
    plt.plot([angle,angle],[np.min(C),np.max(C)])
    #"""
    """
    import matplotlib.pyplot as plt
    x=np.arange(len(top))
    plt.figure()
    plt.plot(x,top)
    plt.plot(x+(C.argmax()-(len(C)-1)/2),bottom)
    plt.title('image angle')
    #"""
    return angle

def init_process(profile, mode):
    """
    Process the initial profile
    
    Parameters
    ----------
    profile:  1d array
        Profile to analyse 
    mode: string
        'none':
            Nothing
        'gaussian':
            Return a gaussian fit
        'tails':
            Remove the tails
        'gfilter':
            Apply a gaussian filter of 2 px std
    Returns
    -------
    profile: 1d array
        the processed profile
    
    """
    profile=np.array(profile)
    if mode == 'none':
        return profile
    elif mode == 'gfilter':
        return gfilter(profile,2)
    elif mode == 'gaussian' or mode == 'tails':
        Y=profile
        X=np.arange(len(Y))
        valid=Y>.5*Y.max()
        gauss=np.exp(np.poly1d(np.polyfit(X[valid],np.log(Y[valid]),2))(X))
        if mode=='gaussian':
            return gauss
        remove=gauss<.01*gauss.max()
        profile[remove]=0
        return profile

def get_fax(profiles):
    """
    returns a faxed verion of the profiles for easier plotting
    
    Parameters
    ----------
    profiles:  2d array
        List of profiles
        
    Returns
    -------
    profiles: 1d array
        The faxed profiles
    
    """
    return np.ravel(np.concatenate(
            (profiles,np.zeros((np.shape(profiles)[0],1))*np.nan),axis=1))
    
def get_edge(profile):
    """Get the largest edge in the profile
    
    Parameters
    ----------
    profile:  1d array
        profile to analyse
    Returns
    -------
    edgePos: float
        The edge position
    """
    e=np.abs(np.diff(gfilter(profile,2)))
    valid=slice(np.argmax(e)-3,np.argmax(e)+4)
    X=np.arange(len(e))+.5
    X=X[valid]
    Y=np.log(e[valid])
    coeff=np.polyfit(X,Y,2)
    edgePos=-coeff[1]/(2*coeff[0])
    return edgePos

def get_profiles(scans, Npix, orientation=None, *, 
                 offset_edge_idx =None, offset=0):
    """Extract profiles from scans
    
    Parameters
    ----------
    scans:  2d array
        sacns to analyse
    Npix:   integer
        number of pixels in a profile
    orientation: 1d array
        Orientation of each scan (Positive or negative)
    offset_edge_idx: integer
        Index of a profile containing an edge and a maximum to detect offset
    offset: integer
        Manual offset
        
    Returns
    -------
    profiles: 1d array
        The profiles
    """
    
    #Init return
    profiles=np.empty((scans.shape[0],Npix))
    scans=np.array(scans)
    if offset_edge_idx is not None and offset_edge_idx<0:
        offset_edge_idx=len(scans)+offset_edge_idx
    
    #Straighten scans
    if orientation is not None:
        for s,o in zip(scans,orientation):
            if o<0:
                s[:]=s[::-1]
    
    # get the offset if needed
    if offset_edge_idx is not None:
        offset_scan=scans[offset_edge_idx]
        cent=center(offset_scan)
        edge=get_edge(offset_scan)
        offset=np.abs(cent-edge)-Npix/2
        edgeside=1
        if edge>cent:
            edgeside=-1
    
    #For each scan
    for i,s in enumerate(scans):
        #Get the mid point
        if offset_edge_idx is None:
            mid=center(s)-offset
        else:
            if i<offset_edge_idx:
                mid=center(s)-edgeside*offset
            else:
                mid=get_edge(s)+edgeside*Npix/2
        #First position
        amin=int(mid-Npix/2)
        #If pixels missings:
        if amin<0 or amin>len(s)-Npix:
            warnings.warn("Missing pixels, scan not large enough", 
                          RuntimeWarning)
            while amin>len(s)-Npix:
                s=np.append(s,s[-1])
            while amin<0:
                amin+=1
                s=np.append(s[0],s)
        #Get profile
        profiles[i]=s[amin:amin+Npix]
        
    return profiles
