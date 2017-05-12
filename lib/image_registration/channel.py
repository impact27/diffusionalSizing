# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:53:44 2016

@author: quentinpeter

This covers an alternative method for image registration aimed at 
simple channels

Copyright (C) 2016  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#imports
from numpy.fft import irfft
from scipy.ndimage.filters import gaussian_filter
import math
import numpy as np
from . import image as reg
import cv2

def channel_width(im, chanangle=None, *, chanapproxangle=None,
                  isccsedge=False):
    """Get an estimation of the channel width. 
    
    Parameters:
    -----------
    im: 2d array
        The channel image
    chanangle: number, optional
        The angle of the channel (None if unknown)
    chanapproxangle: number, optional
        If chanangle is None, the approximate channel angle
    isccsedge: boolean, default False
        Set to True if im is the dft of egde. 
               False if it is an image of a channel.
     
    Returns:
    --------
    width: number
        The channel width
    angle: number
        The corresponding angle
                      
    Notes:
    ------

    This function assumes two parallel lines along angle chanangle.
    The perpendicular line in the fourrier plane will have a mark of this,
    under the form of an oscillation at some frequency corresponding
    to the distance between the two parallel lines.
    This can be extracted by another fft.  
    
    This second fft might have large components at low frequency, 
    So the first few frequencies are neglected.
    The threshold is the first position below mean
    
    If the chanangle is not specified, the direction with higher 
    contribution will be picked.
    
    If chanapproxangle is given, only angles close to this angle
    are taken into account
    """
    #check input is numpy array
    im=np.asarray(im)
    
    #Compute the dft if it is not already done
    if not isccsedge:
        im= reg.dft_optsize(np.float32(edge(im)))
    
    #save the truesize for later use
    truesize=im.shape
    
    #get centered magnitude squared (This changes the size)
    im = reg.centered_mag_sq_ccs(im)
    
    #if the channel direction is not given, deduce it from channel_angle
    if chanangle is None:
        chanangle = channel_angle(im,isshiftdftedge=True,
                                  chanapproxangle=chanapproxangle,
                                  truesize=truesize)
    
    #get vector perpendicular to angle
    fdir=np.asarray([math.cos(chanangle),-math.sin(chanangle)])#y,x = 0,1
    
    #need to be in the RHS of the cadran for rfft
    if fdir[1]<0:
        fdir*=-1    
    
    #get center of shifted fft
    center=np.asarray([im.shape[0]//2,0])
    
    #get size
    shape=np.asarray([im.shape[0]//2,im.shape[1]])
    
    #get evenly spaced positions between 0 and 1 (not included) 
    pos=np.r_[:1:(shape.min()+1)*1j][:-1]#should be replaced with linspace
              
    #get index of a line of length 1 in normalized units from center 
    #in direction of chdir
    idx=((fdir*shape)[:,np.newaxis].dot(pos[np.newaxis])
         +center[:,np.newaxis])
    
    #get the line
    idx=np.float32(idx)
    f=cv2.remap(np.float32(im),idx[1,:],idx[0,:],cv2.INTER_LINEAR)
    f=np.squeeze(f)
    
    #The central line of the fft will have a periodic feature for parallel
    #lines which we can detect with fft
    f=abs(irfft(f**2))
    #filter to avoid "interferences"
    f=gaussian_filter(f,1)
    #the offset is determined by the first pixel below mean
    wmin=np.nonzero(f-f.mean()<0)[0][0]
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(f,'x')
    plt.plot([wmin,wmin],[0,f.max()])
    plt.plot([0,500],[f.mean()+3*f.std(),f.mean()+3*f.std()])
    #"""
    
     #find max excluding the first few points
    ret=reg.get_peak_pos(f[wmin:f.size//2])
    
    #return max and corresponding angle
    return (wmin+ret), chanangle
    
def channel_angle(im,chanapproxangle=None, *, isshiftdftedge=False,
                  truesize=None):
    """Extract the channel angle from the rfft
    
    Parameters:
    -----------
    im: 2d array
        The channel image
    chanapproxangle: number, optional
        If not None, an approximation of the result
    isshiftdftedge: boolean, default False
        If The image has already been treated:
        (edge, dft, fftshift), set to True
    truesize: 2 numbers, required if isshiftdftedge is True
        The true size of the image
              
    Returns:
    --------
    angle: number
        The channel angle
    
    
    """
    im=np.asarray(im)
    #Compute edge
    if not isshiftdftedge:
        im=edge(im)
    return reg.orientation_angle(im,isshiftdft=isshiftdftedge,
                                 approxangle=chanapproxangle,
                                 truesize=truesize)
    
def Scharr_edge(im, blurRadius=10, imblur=None):
    """Extract the edges using Scharr kernel (Sobel optimized for rotation
    invariance)
    
    Parameters:
    -----------
    im: 2d array
        The image
    blurRadius: number, default 10
        The gaussian blur raduis (The kernel has size 2*blurRadius+1)
    imblur: 2d array, OUT
        If not None, will be fille with blurred image
    
    Returns:
    --------
    out: 2d array
        The edges of the images computed with the Scharr algorithm
    
    """
    im=np.asarray(im,dtype='float32')
    blurRadius=2*blurRadius+1
    im=cv2.GaussianBlur(im,(blurRadius,blurRadius),0)
    Gx=cv2.Scharr(im,-1,0,1)
    Gy=cv2.Scharr(im,-1,1,0)
    ret=cv2.magnitude(Gx,Gy)
    if imblur is not None and imblur.shape == im.shape:
        imblur[:,:]=im
    return ret
    
def edge(im):
    """Extract the edges of an image
    
    Parameters:
    -----------
    im: 2d array
        The image
    
    Returns:
    --------
    out: 2d array
        The edges of the images computed with the Canny algorithm
    
    Notes:
    ------
    This scale the image to be used with Canny from OpenCV    
    """
    #map the 16bit image into 8bit   
    e0=cv2.Canny(uint8sc(im),100,200)
    return e0
    
def register_channel(im0,im1,scale=None,ch0angle=None, chanapproxangle=None):
    """Register the images assuming they are channels
    
    Parameters:
    -----------
    im0: 2d array
        The first image
    im1: 2d array
        The second image
    scale: number, optional
        The scale difference if known
    ch0angle: number, optional
        The angle of the channel in the first image if known 
    chanapproxangle: number, optional
        The approximate angle for both images if known
    
    Returns:
    --------
    angle: number
        The angle difference
    scale: number
        The scale difference
    [y, x]: 2 numbers
        The offset
    e2: 2d array
        The second image rotated and translated for performances reasons
    
    """
    im0=np.asarray(im0)
    im1=np.asarray(im1)
    #extract the channels edges
    e0=edge(im0)
    e1=edge(im1)
    fe0, fe1=reg.dft_optsize_same(np.float32(e0),np.float32(e1))
    
    #compute the angle and channel width of biggest angular feature
    w0,a0=channel_width(fe0,isccsedge=True, chanapproxangle=chanapproxangle)
    w1,a1=channel_width(fe1,isccsedge=True, chanapproxangle=chanapproxangle)
    
    #get angle diff
    angle=reg.clamp_angle(a0-a1)
    if ch0angle is not None:
        a0=ch0angle
        a1=a0-angle 
        
    #if the scale is unknown, ratio of the channels
    if scale is None:
        scale=w1/w0
    #scale and rotate
    e2=reg.rotate_scale(e1,angle,scale)
    #get edge from scaled and rotated im1
    fe2=reg.dft_optsize(np.float32(e2),shape=fe0.shape)
    #find offset
    y,x=reg.find_shift_dft(fe0,fe2, isccs=True)
    #return all infos
    return angle, scale, [y, x], e2
    
def uint8sc(im):
    """Scale the image to uint8
    
    Parameters:
    -----------
    im: 2d array
        The image
    
    Returns:
    --------
    im: 2d array (dtype uint8)
        The scaled image to uint8
    """
    im=np.asarray(im)
    immin=im.min()
    immax=im.max()
    imrange=immax-immin
    return cv2.convertScaleAbs(im-immin, alpha=255/imrange)
