# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:26:20 2017

@author: quentinpeter
"""
import numpy as np
import background_rm as rmbg
import image_registration.image as ir
import image_registration.channel as cr
import diffusion_device.profile as dp
import scipy
import matplotlib.image as mpimg
import warnings
import cv2
from scipy import interpolate
warnings.filterwarnings('ignore', 'Mean of empty slice',RuntimeWarning)

def size_images(images,Q,Wz,pixsize,readingpos=None,Rs=None,chanWidth=300e-6,*,
                Zgrid=11,ignore=10e-6,normalize_profiles=True,initmode='none',
                data_dict=None,rebin=2,):
    """
    Get the hydrodynamic radius from the images
    
    Parameters
    ----------
    images: 1d list of images or file name OR 2x 1d list
        If this is a string, it will be treated like a path
        If one list, treated like regular fluorescence images
        If two list, treated like images and backgrounds
    Q: float
        Flow rate in [ul/h]
    Wz: float
        Height of the channel in [m]
    pixsize: float
        Pixel size in [m]
    readingpos: 1d float array, defaults None
        Position at which the images are taken. If None, take the defaults
    Rs: 1d array, defaults None
        Hydrodimamic radii to simulate in [m].
        If None: between .5 and 10 nm
    chanWidth: float, default 300e-6
        The channel width in [m]
    Zgrid: int, defaults 11
        Number of Z slices
    ignore: float, defaults 10e-6
        Distance to sides to ignore
    normalize_profiles: Bool, defaults True
        Should the profiles be normalized?
    initmode: str, defaults 'none'
        The processing mode for the initial profile (See profiles.py)
    data_dict: dict, defaults None
        Output to get the profiles and fits
    rebin: int, defaults 2
        Rebin factor to speed up code
        
    Returns
    -------
    r: float
        Radius in [m]
    
    """
    
    #Check images is numpy array
    images=np.asarray(images)
    
    #Fill missing arguments
    if readingpos is None:
        readingpos=defaultReading12Pos()
    if Rs is None:
        Rs=np.arange(.5,10,.5)*1e-9
    
    #load images if string
    if images.dtype.type==np.str_:
        if len(np.shape(images))==1:
            images=np.asarray(
                    [mpimg.imread(im) for im in images])
        elif len(np.shape(images))==2:
            images=np.asarray(
                    [[mpimg.imread(im) for im in ims] for ims in images])
    #Get flat images
    if len(np.shape(images))==3:
        #Single images
        flatimages=np.asarray(
                [flat_image(im,pixsize, chanWidth) 
                for im in images])
    elif len(np.shape(images))==4 and np.shape(images)[0]==2:
        #images and background
        flatimages=np.asarray(
                [remove_bg(im,bg,pixsize, chanWidth) 
                for im,bg in zip(images[0],images[1])])
    
    if rebin>1:   
        size=tuple(np.array(np.shape(flatimages)[1:])//rebin)
        flatimages=np.array([cv2.resize(im,size,interpolation=cv2.INTER_AREA)
                                for im in flatimages])
        pixsize*=rebin  
      
    #get profiles
    profiles=np.asarray(
            [extract_profile(fim,pixsize, chanWidth) for fim in flatimages])
    
    if data_dict is not None:
        data_dict['pixsize']=pixsize
        data_dict['profiles']=profiles  

    return dp.size_profiles(profiles,Q,Wz,pixsize,readingpos,Rs,
                  initmode=initmode,normalize_profiles=normalize_profiles,
                  Zgrid=Zgrid, ignore=ignore,data_dict=data_dict)
    

def remove_bg(im,bg, pixsize, chanWidth=300e-6):
    """
    Remove background from image
    
    Parameters
    ----------
    im: 2d array
        image 
    bg: 2d array
        background
    pixsize: float
        pixel size in [m]
    chanWidth: float, defaults 300e-6
        channel width  in [m]
        
    Returns
    -------
    im: 2d array
        The processed image
    
    """
    im=np.array(im,dtype=float)
    bg=np.array(bg,dtype=float)
    #remove dust peaks on images
    bg[rmbg.getPeaks(bg, maxsize=50*50)]=np.nan
    im[rmbg.getPeaks(im, maxsize=50*50)]=np.nan  
    
    #Get the X positions (perpendicular to alignent axis) and check wide enough
    X=np.arange(im.shape[1])*pixsize
    assert(1.2*chanWidth<X[-1])
    
    #Get the approximate expected channel position
    channel=np.absolute(X-X[-1]/2)<.6*chanWidth
    
    #Create mask to ignore channel when flattening image
    mask=np.ones(im.shape,dtype=bool)
    mask[:,channel]=False
    
    #Get data
    return rmbg.remove_curve_background(im,bg,maskim=mask)

def flat_image(im, pixsize, chanWidth=300e-6):
    """
    Flatten the image
    
    Parameters
    ----------
    im: 2d array
        image 
    pixsize: float
        pixel size in [m]
    chanWidth: float, defaults 300e-6
        channel width  in [m]
        
    Returns
    -------
    im: 2d array
        The flattened image
    """
    
    im=np.asarray(im,dtype=float)
    #remove peaks
    im[rmbg.getPeaks(im, maxsize=20*20)]=np.nan
    #straighten
    angle=dp.image_angle(im-np.nanmedian(im))
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    
    #Get center
    prof=np.nanmean(im,0)
    flatprof=prof-np.nanmedian(prof)
    flatprof[np.isnan(flatprof)]=0
    x=np.arange(len(prof))-dp.center(flatprof)
    x=x*pixsize
    
    #Create mask
    channel=np.abs(x)<chanWidth/2
    mask=np.ones(np.shape(im))
    mask[:,channel]=0
    
    #Flatten
    im=im/rmbg.polyfit2d(im,mask=mask)-1
    
    """
    from matplotlib.pyplot import figure, imshow,plot
    figure()
    imshow(im)
    imshow(mask,alpha=.5,cmap='Reds')
#    plot(x,flatprof)
#    plot(x,np.correlate(flatprof,flatprof[::-1],mode='same'))
    #"""
    
    return im

def extract_profile(flatim, pixsize, chanWidth=300e-6,*,reflatten=True,ignore=10):
    """
    Get profile from a flat image
    
    Parameters
    ----------
    flatim: 2d array
        flat image 
    pixsize: float
        pixel size in [m]
    chanWidth: float, defaults 300e-6
        channel width  in [m]
    reflatten: Bool, defaults True
        Should we reflatten the profile?
    ignore: int, defaults 10
        The number of pixels to ignore if reflattening
        
    Returns
    -------
    im: 2d array
        The flattened image
    """
    
    #Orientate
    flatim=ir.rotate_scale(flatim,-dp.image_angle(flatim)
                            ,1, borderValue=np.nan)
    #get profile
    prof=np.nanmean(flatim,0)
    
    #Center X
    X=np.arange(len(prof))*pixsize
    center=dp.center(prof)*pixsize
    inchannel=np.abs(X-center)<.45*chanWidth
    X=X-(dp.center(prof[inchannel])+np.argmax(inchannel))*pixsize
    
    #get what is out
    out=np.logical_and(np.abs(X)>.55*chanWidth,np.isfinite(prof))
    
    if reflatten:
        #fit ignoring extreme 10 pix
        fit=np.polyfit(X[out][ignore:-ignore],prof[out][ignore:-ignore],2)
        bgfit=fit[0]*X**2+fit[1]*X+fit[2]
        
        #Flatten the profile
        prof=(prof+1)/(bgfit+1)-1

    #We restrict the profile to channel width - widthcut
    Npix=int(chanWidth//pixsize)+1
    
    Xc=np.arange(Npix)-(Npix-1)/2
    Xc*=pixsize
    
    finterp=interpolate.interp1d(X, prof,bounds_error=False,fill_value=0)
    """
    from matplotlib.pyplot import figure, imshow,plot
    figure()
    plot(X,prof)
    #"""  
    return finterp(Xc)
    
    
    """
    from matplotlib.pyplot import figure, imshow,plot
    figure()
    imshow(flatim)
    plot([c-Npix//2,c-Npix//2],[5,np.shape(flatim)[0]-5],'r')
    plot([c+Npix//2,c+Npix//2],[5,np.shape(flatim)[0]-5],'r')
    figure()
    pr=np.nanmean(flatim,0)
    plot(pr)
    plot([c-Npix//2,c-Npix//2],[np.nanmin(pr),np.nanmax(pr)],'r')
    plot([c+Npix//2,c+Npix//2],[np.nanmin(pr),np.nanmax(pr)],'r')
    #"""  
    
#    return prof[channel]
 
def defaultReading12Pos():
    '''
    Get the default reading positions for the 12 points diffusion device
    
    Returns
    -------
    readingPos: 1d array
        The reading positions
    '''
    return np.array([0,
                     3.5,
                     5.3,
                     8.6,
                     10.3,
                     18.6,
                     20.4,
                     28.6,
                     30.4,
                     58.7,
                     60.5,
                     88.7,
                     90.5])*1e-3    
    
def outChannelMask(im, chAngle=0):
    """Creates a mask that excludes the channel
    
    Parameters
    ----------
    im: 2d array
        The image
    chAngle: number
        The angle of the channel in radians
    
    Returns
    -------
    mask: 2d array
        the mask excluding the channel
        
    Notes
    -----
    The channel should be clear(ish) on the image. 
    The angle should be aligned with the channel
    

    """
    im=np.array(im,dtype='float32')
    #Remove clear dust
    mask=rmbg.backgroundMask(im, nstd=6)
    im[~mask]=np.nan
    
    #get edge
    scharr=cr.Scharr_edge(im)
    #Orientate image along x if not done
    if chAngle !=0:
        scharr= ir.rotate_scale(scharr, -chAngle,1,np.nan)
        
    #get profile
    prof=np.nanmean(scharr,1)
    #get threshold
    threshold=np.nanmean(prof)+3*np.nanstd(prof)
    mprof=prof>threshold
    edgeargs=np.flatnonzero(mprof)
    
    if edgeargs.size > 2:
        mask=np.zeros(im.shape)
        mask[edgeargs[0]-5:edgeargs[-1]+5,:]=2
        if chAngle !=0:
            mask= ir.rotate_scale(mask, chAngle,1,np.nan)
        mask=np.logical_and(mask<1, np.isfinite(im))
    else:
        mask= None
    return mask
    
def outGaussianBeamMask(data, chAngle=0):
    """
    get the outside of the channel from a gaussian fit
    
    Parameters
    ----------
    data: 2d array
        The image
    chAngle: number
        The angle of the channel in radians
    
    Returns
    -------
    mask: 2d array
        the mask excluding the channel
    
    """
    data=np.asarray(data)
    
    #Filter to be used
    gfilter=scipy.ndimage.filters.gaussian_filter1d
    
    #get profile
    if chAngle!=0:
        data=ir.rotate_scale(data, -chAngle,1,np.nan)
    profile=np.nanmean(data,1)
    
    #guess position of max
    amax= profile.size//2
    
    #get X and Y
    X0=np.arange(profile.size)-amax
    Y0=profile
    
    #The cutting values are when the profiles goes below zero
    rlim=np.flatnonzero(np.logical_and(Y0<0,X0>0))[0]
    llim=np.flatnonzero(np.logical_and(Y0<0,X0<0))[-1]
    
    #We can now detect the true center
    fil=gfilter(profile,21)
    X0=X0-X0[np.nanargmax(fil[llim:rlim])]-llim
    
    #restrict to the correct limits
    X=X0[llim:rlim]
    Y=Y0[llim:rlim]-np.nanmin(Y0)
    
    #Fit the log, which should be a parabola
    c=np.polyfit(X,np.log(Y),2)
    
    #Deduce the variance
    var=-1/(2*c[0])
    
    #compute the limits (3std, restricted to half the image)
    mean=np.nanargmax(fil[llim:rlim])+llim
    dist=int(3*np.sqrt(var))
    if dist > profile.size//4:
        dist = profile.size//4
    llim=mean-dist
    if llim < 0:
        return None
    rlim=mean+dist
    if rlim>profile.size:
        return None
    
    #get mask
    mask=np.ones(data.shape)
    
    if chAngle!=0:
        idx=np.indices(mask.shape)
        
        
        idx[1]-=mask.shape[1]//2
        idx[0]-=mask.shape[0]//2
        X=np.cos(chAngle)*idx[1]+np.sin(chAngle)*idx[0]
        Y=np.cos(chAngle)*idx[0]-np.sin(chAngle)*idx[1]
        
        mask[np.abs(Y-mean+mask.shape[0]//2)<dist]=0
        
    else:    
        mask[llim:rlim,:]=0
    
    #mask=np.logical_and(mask>.5, np.isfinite(data))
    mask=mask>.5
    return mask
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    #plot profile and fit
    valmax=np.nanmax(Y)
    plt.plot(X0,Y0)
    plt.plot(X0,valmax*np.exp(-(X0**2)/(2*var))+np.nanmin(Y0))
    plt.plot([llim-mean,llim-mean],[np.nanmin(Y0),np.nanmax(Y0)],'r')
    plt.plot([rlim-mean,rlim-mean],[np.nanmin(Y0),np.nanmax(Y0)],'r')
    #"""

