# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:21:01 2017

@author: quentinpeter
"""
import numpy as np
import matplotlib.image as mpimg
from . import bright, background
import diffusion_device.profile as dp

def defaultReadingPos(startpos=400e-6, isFolded=True):
    '''
    Get the default reading positions for the 4 points diffusion device
    
    Parameters
    ----------
    startpos: float, default 400e-6 
        The center of the image, relative to the first turn [m]
    isFolded: Bool, default True
        If this is the folded or the straight device
    
    Returns
    -------
    readingPos: 1d array
        The reading positions
    '''
#    return np.array([  4183, 21446, 55879])*1e-6 #Kadi
#    return np.array([  3738, 21096, 55374])*1e-6 #Electrophoresis
    if isFolded:
        return np.array([0,4556e-6-2*startpos, 
                         21953e-6, 
                         47100e-6-2*startpos]) #folded device
    else:
        return np.array([0,4532e-6-2*startpos, 
                         21128e-6, 
                         56214e-6-2*startpos]) #folded device
    



def size_image(im,Q,Wz,Wy,readingpos=None,Rs=None,*,
                Zgrid=11,ignore=5e-6,normalize_profiles=True,initmode='none',
                data_dict=None, fit_position_number=None,imSlice=None,
                flatten=False):
    
    """
    Get the hydrodynamic radius from the images
    
    Parameters
    ----------
    im: 2d image or file name OR 2x 2d images
        If this is a string, it will be treated like a path
        If one image, treated like regular fluorescence image
        If two images, treated like image and background
    Q: float
        Flow rate in [ul/h]
    Wz: float
        Height of the channel in [m]
    Wy: float
        Width of the channel in [m]
    readingpos: 1d float array, defaults None
        Position at which the images are taken. If None, take the defaults
    Rs: 1d array, defaults None
        Hydrodimamic radii to simulate in [m].
        If None: between .5 and 10 nm
    Zgrid: int, defaults 11
        Number of Z slices
    ignore: float, defaults 5e-6
        Distance to sides to ignore
    normalize_profiles: Bool, defaults True
        Should the profiles be normalized?
    initmode: str, defaults 'none'
        The processing mode for the initial profile (See profiles.py)
    data_dict: dict, defaults None
        Output to get the profiles and fits
    fit_position_number: 1d list
        Positions to use in the fit
    imSlice: slice
        slice of the image to use
    flatten: Bool, defaut False
        (Bright field only) Should the image be flattened?
        
    Returns
    -------
    r: float
        Radius in [m]
    
    """
    
    #Check images is numpy array
    im=np.asarray(im)
    
    #Fill missing arguments
    if readingpos is None:
        readingpos=defaultReadingPos()
    if Rs is None:
        Rs=np.arange(.5,10,.5)*1e-9
    
    #load images if string
    if im.dtype.type==np.str_:
        if len(np.shape(im))==0:
            im=mpimg.imread(str(im))
        elif len(np.shape(im))==1:
            im=np.asarray([mpimg.imread(fn) for fn in im])
            
    try:
        #get profiles
        if len(np.shape(im))==2:
            #Single image
            profiles=bright.extract_profiles(im,imSlice,flatten=flatten)
        elif len(np.shape(im))==3 and np.shape(im)[0]==2:
            #images and background
            profiles= background.extract_profiles(im[0],im[1],imSlice)
    except RuntimeError as error:
        if error.args[0]=="Can't get image infos":
            return np.nan
        raise
    

    pixsize=Wy/np.shape(profiles)[1]
    if data_dict is not None:
        data_dict['pixsize']=pixsize
        data_dict['profiles']=profiles
    return dp.size_profiles(profiles,Q,Wz,pixsize,readingpos,Rs,
                  initmode=initmode,normalize_profiles=normalize_profiles,
                  Zgrid=Zgrid, ignore=ignore,data_dict=data_dict,
                  fit_position_number=fit_position_number)
