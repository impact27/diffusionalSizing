# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:18:33 2017

@author: quentinpeter
"""
import numpy as np
import scipy.ndimage
gfilter=scipy.ndimage.filters.gaussian_filter1d
from scipy.ndimage.filters import maximum_filter1d
import diffusion_device.profile as dp
import background_rm as rmbg
import image_registration.image as ir
from scipy import interpolate


def image_infos(im):
    """
    Get the image angle, channel width, proteind offset, and origin
    
    Parameters
    ----------
    im: 2d array
        The image
        
    Returns
    -------
    dict: dictionnary
        dictionnary containing infos
    
    """
    imflat=im
    #Detect Angle
    angle=dp.image_angle(imflat)
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    #Get channels infos
    w,a,origin=straight_image_infos(im)
    
    retdict={
            'angle':angle,
            'origin':origin,
            'width':w,
            'offset':a}
    return retdict

def straight_image_infos(im, Nprofs=4):
    """
    Get the channel width, proteind offset, and origin from a straight image
    
    Parameters
    ----------
    im: 2d array
        The image
        
    Returns
    -------
    w: float
        Channel width in pixels
    a: float
        offset of the proteins in the channel
    origin: float
        Position of the first channel center
    
    """
    width_pixels=np.shape(im)[1]//10
    
    profiles=np.nanmean(im-np.nanmedian(im),0)
    
    #Find max positions
    fprof=gfilter(profiles,3)
    maxs=np.where(maximum_filter1d(fprof,width_pixels)==fprof)[0]
    maxs=maxs[np.logical_and(maxs>15,maxs<len(fprof)-15)]
    maxs=maxs[np.argsort(fprof[maxs])[-Nprofs:]][::-1]
#    from matplotlib.pyplot import figure, show, plot, imshow
#    figure()
#    plot(fprof)
#    plot(maximum_filter1d(fprof,100))
#    for m in maxs:
#        plot([m,m],[0,np.nanmax(fprof)])

    
    if len(maxs)<Nprofs:
        raise RuntimeError("Can't get image infos")   

    profiles -=np.nanmin(profiles)
        
    maxs=np.asarray(maxs,dtype=float)
    for i,amax in enumerate(maxs):
        amax=int(amax)
        y=np.log(profiles[amax-10:amax+10])
        x=np.arange(len(y))
        coeff=np.polyfit(x[np.isfinite(y)],y[np.isfinite(y)],2)
        maxs[i]=-coeff[1]/(2*coeff[0])-10+amax
     
    maxs=np.sort(maxs)
    
    if maxs[0]<0 or maxs[-1]>len(profiles):
        raise RuntimeError("Can't get image infos")
        
    if fprof[int(maxs[0])]>fprof[int(maxs[-1])]:
        #Deduce relevant parameters
        w=(maxs[2]-maxs[0])/4
        a=w+(maxs[0]-maxs[1])/2
        origin=maxs[0]-a   
        lastdist=maxs[3]-(origin+6*w-a)
        
    else:
        #Deduce relevant parameters
        w=(maxs[3]-maxs[1])/4
        a=w+(maxs[2]-maxs[3])/2
        origin=maxs[3]+a-6*w   
        lastdist=maxs[0]-(origin+a)
        
    
    assert w>0, 'Something went wrong while analysing the images'
    #if position 4 is remotely correct, return infos
    if (    np.abs(lastdist)>w/2 #Last too far
            or np.any(np.isnan((a,w,origin,maxs[3])))#got nans
            or origin -a +6.5*w >len(profiles)#Right side not in
            or origin+a-.5*w<0):#left side not in
        raise RuntimeError("Can't get image infos")
    return w,a,origin
    

def flat_image(im,frac=.7,infosOut=None, subtract=False):
    """
    Flatten input images
    
    Parameters
    ----------
    im: 2d array
        The image
    frac: float
        fraction of the profile taken by fluorescence from channels
    infosOut: dict, defaults None
        dictionnary containing the return value of straight_image_infos
    subtract: Bool
        Should the shape be subtracted instead of divided
        
    Returns
    -------
    im: 2d array
        The flattened image
    
    """
    #Detect Angle
    angle=dp.image_angle(im-np.median(im))
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    #Get channels infos
    w,a,origin=straight_image_infos(im)
    #get mask
    mask=np.ones(np.shape(im)[1])
    for i in range(4):
        amin=origin+2*i*w-frac*w
        amax=origin+2*i*w+frac*w
        mask[int(amin):int(amax)]=0
    mask=mask>0
    mask=np.tile(mask[None,:],(np.shape(im)[0],1))
    #Flatten
    if not subtract:
        im=im/rmbg.polyfit2d(im,mask=mask)-1
    else:
        im=im-rmbg.polyfit2d(im,mask=mask)
#        import matplotlib.pyplot as plt
#        plt.figure()
#        plt.imshow(rmbg.polyfit2d(im,mask=mask))
#        plt.colorbar()
#        plt.figure()
#        plt.imshow(im)
#        plt.imshow(mask,alpha=.5)
    if infosOut is not None:
        infosOut['infos']=(w,a,origin)
    return im
    
def extract_profiles_flatim(im,infos):
    '''
    Extract profiles from flat image
    
    Parameters
    ----------
    im: 2d array
        The flat image
    infos: dict
        dictionnary containing the return value of straight_image_infos
        
    Returns
    -------
    profiles: 2d array
        The four profiles
    '''
    #Find positions
    w,a,origin=infos
    image_profile=np.nanmean(im,0)
    
    #Extract one by one
    Npix=int(np.round(w))
    profiles=np.zeros((4,Npix))
    
    for i in range(4):   
        X=np.arange(len(image_profile))-(origin+2*i*w)        
        Xc=np.arange(Npix)-(Npix-1)/2
        finterp=interpolate.interp1d(X, image_profile)
        protoprof = finterp(Xc)
        #switch if uneven
        if i%2==1:
            protoprof=protoprof[::-1]
            
        profiles[i]=protoprof
    
    #If image upside down, turn
    if profiles[-1].max()>profiles[0].max():
        profiles=profiles[::-1] 
        
    """
    from matplotlib.pyplot import plot, figure, imshow
    figure()
    imshow(im)
    figure()
    plot(image_profile)
    #"""
    return profiles

def extract_profiles(im, imSlice=None,flatten=False):
    '''
    Extract profiles from image
    
    Parameters
    ----------
    im: 2d array
        The flat image
    imSlice: slice
        Slice of the image to consider
    flatten: Bool, Defaults False
        Should the image be flatten
        
    Returns
    -------
    profiles: 2d array
        The four profiles
    '''
    im=np.asarray(im)
    if imSlice is not None:
        im=im[imSlice]
    infos={}
    if flatten:
        im=flat_image(im,infosOut=infos)
    angle=dp.image_angle(im)
    im=ir.rotate_scale(im,-angle,1,borderValue=np.nan)
    if not flatten:
        infos['infos']=straight_image_infos(im)
    """
    profiles0=extract_profiles_flatim(im[:100],infos['infos'])
    for p in profiles0:
        p/=np.mean(p)
    profiles1=extract_profiles_flatim(im[-100:],infos['infos'])
    for p in profiles1:
        p/=np.mean(p)
    from matplotlib.pyplot import plot, figure, imshow
    figure()
    plot(np.ravel(profiles0))
    plot(np.ravel(profiles1))
    #"""
    profiles=extract_profiles_flatim(im,infos['infos'])
    return profiles
