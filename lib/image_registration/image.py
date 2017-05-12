# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:15:48 2016

@author: Quentin Peter

This module does image registration. 
    
The scale difference should be reasonable (~2x)
    
If you crop the images to reasonable size, the algorithm is much faster:
eg:
511x511   : 0.058s
512x512   : 0.058s
513x513   : 0.066s /!\ 15% increase!
1024x1024 : 0.34s
4096x4096 : 9s
    
based on:
    
An FFT-Based Technique for Translation, Rotation, and Scale-Invariant
 Image Registration
B. Srinivasa Reddy and B. N. Chatterji

Works with Python 3.5

The input is casted to float32. If you need more precision, you can always 
use these functions to detect the rotation and then rotate yourself with scipy
(slower but keep the precision)

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
#import libraries
import numpy as np
import cv2
from scipy.optimize import curve_fit # /!\ SLOW! avoid
from scipy.ndimage.measurements import label


######################
#High level functions#
######################


def register_images(im0,im1,*,rmMean=True):
    """Finds the rotation, scaling and translation of im1 relative to im0
    
    Parameters
    ----------
    im0: First image
    im1: Second image
    rmMean: Set to true to remove the mean (Default)
    
    Returns
    -------
    angle: The angle difference
    scale: The scale difference
    [y, x]: The offset
    im2: The rotated and translated second image
    
    Notes
    -----
    The algorithm uses gaussian fit for subpixel precision.
    
    The best case would be to have two squares images of the same size.
    The algorithm is faster if the size is a power of 2.
    """
    #sanitize input
    im0=np.asarray(im0,dtype=np.float32)
    im1=np.asarray(im1,dtype=np.float32)
    if rmMean:
        #remove mean
        im0= im0 - im0.mean()
        im1= im1 - im1.mean()
    #Compute DFT (THe images are resized to the same size)
    f0, f1=dft_optsize_same(im0,im1)
    #Get rotation and scale
    angle, scale = find_rotation_scale(f0,f1, isccs=True)
    #apply rotation and scale
    im2=rotate_scale(im1,angle,scale)
    f2=dft_optsize(im2,shape=f0.shape)
    #Find offset
    y,x=find_shift_dft(f0,f2, isccs=True)
    return angle, scale, [y, x], im2


def find_rotation_scale(im0,im1,isccs=False):
    """Compares the images and return the best guess for the rotation angle,
    and scale difference.
    
    Parameters
    ----------
    im0: 2d array
        First image
    im1: 2d array
        Second image
    isccs: boolean, default False
        Set to True if the images are alredy DFT and in CCS representation
    
    Returns
    -------
    angle: number
        The angle difference
    scale: number
        The scale difference
    
    Notes
    -----
    Uses find_shift_dft
    """
    #sanitize input
    im0=np.asarray(im0,dtype=np.float32)
    im1=np.asarray(im1,dtype=np.float32)
    truesize=None
    
    #if ccs, convert to shifted dft before giving to polar_fft
    if isccs:
        truesize=im0.shape
        im0=centered_mag_sq_ccs(im0)
        im1=centered_mag_sq_ccs(im1)
    #Get log polar coordinates. choose the log base 
    lp1, log_base=polar_fft(im1, logpolar=True, isshiftdft=isccs,
                            logoutput = True, truesize=truesize)
    lp0, log_base=polar_fft(im0, logpolar=True, isshiftdft=isccs,
                            logoutput = True, truesize=truesize,
                            nangle=lp1.shape[0], radiimax=lp1.shape[1])
    #Find the shift with log of the log-polar images,
    #to compensate for dft intensity repartition
    angle, scale = find_shift_dft(lp0,lp1)
    #get angle in correct units
    angle*=np.pi/lp1.shape[0]
    #get scale in linear units
    scale=log_base ** (scale)
    #return angle and scale
    return angle, scale


def find_shift_dft(im0,im1, isccs=False, subpix=True):
    """Find the shift between two images using the DFT method
    
    Parameters
    ----------
    im0: 2d array
        First image
    im1: 2d array
        Second image
    isccs: Boolean, default false
        Set to True if the images are alredy DFT and in CCS representation
    subpix: boolean, default True
        Set to True (default) if you want subpixel precision
    
    Returns
    -------
    [y, x]: 2 numbers
        The offset
    
    Notes
    -----
    This algorithm detect a shift using the global phase difference of the DFTs
    
    If the images are already DFT and in the CCS format, set isccs to true.
    In that case the images should have the same size.  
    
    If subpix is True, a gaussian fit is used for subpix precision 
    """
    #sanitize input
    im0=np.asarray(im0,dtype=np.float32)
    im1=np.asarray(im1,dtype=np.float32)
    
    #check input
    if not isccs:
        im0,im1=dft_optsize_same(im0,im1)
    else:
        #Work only if the shapes are the same
        assert(im0.shape==im1.shape)
    
    #f0*conj(f1)
    mulSpec=cv2.mulSpectrums(im0,im1,flags=0,conjB=True)
    #norm(f0)*norm(f1)
    normccs=cv2.sqrt(cv2.mulSpectrums(im0,im0,flags=0,conjB=True)*
                       cv2.mulSpectrums(im1,im1,flags=0,conjB=True))
    #compute the inverse DFT    
    xc=cv2.dft(ccs_normalize(mulSpec,normccs),
               flags=cv2.DFT_REAL_OUTPUT|cv2.DFT_INVERSE)
    #Blur xc to remove some noise and improve the subpixel detection
    #workaround as GaussianBlur doesn't work with BORDER_WRAP
    blurRadii=2
    xc = cv2.copyMakeBorder(xc, blurRadii, blurRadii, blurRadii, blurRadii, 
                            borderType=cv2.BORDER_WRAP)
    xc = cv2.GaussianBlur(xc,(2*blurRadii+1,2*blurRadii+1),1.5)
    xc=xc[blurRadii:-blurRadii,blurRadii:-blurRadii]
    #save shape
    shape=np.asarray(xc.shape)
    #find max
    idx=np.asarray(np.unravel_index(np.argmax(xc),shape)) 
    
    """
    from matplotlib import pyplot as plt
    from numpy.fft import fftshift
    plt.figure()
    plt.imshow(fftshift(xc))
    
    #"""
    #plt.imshow(fftshift(xc))
    #print(idx)
    #plt.figure()
#    if toremove:
#        plt.figure(1)
#        l=len(xc[:,0])
#        plt.plot(np.arange(l)/l,xc[:,0])
#        print(l,xc[-1,0])
#        plt.figure(2)
        
    #"""
     
    if subpix:
        #update idx
        idx=np.asarray([get_peak_pos(xc[:,idx[1]],wrap=True),
                        get_peak_pos(xc[idx[0],:],wrap=True)])
    else:
        #restrics to reasonable values
        idx[idx>shape//2]-=shape[idx>shape//2]
    return idx


def find_shift_cc(im0,im1,ylim=None,xlim=None):
    """Finds the best shift between im0 and im1 using cross correlation
    
    Parameters
    ----------
    im0: 2d array
        First image
    im1: 2d array
        Second image
    ylim: 2 numbers, optional
        The y limits of the search (if None full range is searched)
    xlim: 2 numbers, optional
        Ibidem with x
    
    Returns
    -------
    [y, x]: 2 numbers
        The offset
    
    Notes
    -----
    The origin of im1 in the im0 referential is returned
    
    ylim and xlim limit the possible output.
    
    No subpixel precision
    """
    #sanitize input
    im0=np.asarray(im0,dtype=np.float32)
    im1=np.asarray(im1,dtype=np.float32)
    #Remove mean
    im0=im0-im0.mean()
    im1=im1-im1.mean()
    #Save shapes as np array
    shape0=np.asarray(im0.shape)
    shape1=np.asarray(im1.shape)
    
    #Compute the offset and the pad (yleft,yright,xtop,xbottom)
    offset=1-shape1
    pad=np.lib.pad(-offset,(1,1),mode='edge')
    
    #apply limit on padding
    if ylim is not None:
        pad[0]=-ylim[0]
        pad[1]=ylim[1]+(shape1-shape0)[0]
    if xlim is not None:
        pad[2]=-xlim[0]
        pad[3]=xlim[1]+(shape1-shape0)[1]
    
    #pad image
    im0, offset = pad_img(im0,pad)
    #compute Cross correlation matrix
    xc=cv2.matchTemplate(im0,im1,cv2.TM_CCORR)
    #Find maximum of abs (can be anticorrelated)
    idx=np.asarray(np.unravel_index(np.argmax(xc),xc.shape))        
    #Return origin in im0 units
    return idx+offset
    
def combine_images(imgs, register=True):
    """Combine similar images into one to reduce the noise
    
    Parameters
    ----------
    imgs: list of 2d array
        Series of images
    register: Boolean, default False
        True if the images should be register before combination
    
    Returns
    -------
    im: 2d array
        The result image
    
    Notes
    -----
    This is an example of the usage of the library    
    """
    imgs = np.asarray(imgs,dtype="float")
    if register:
        for i in range(1,imgs.shape[0]):
            ret =register_images(imgs[0,:,:],imgs[i,:,:])
            imgs[i,:,:]=rotate_scale_shift(imgs[i,:,:],*ret[:3], np.nan)
    return np.mean(imgs,0)    
    
########################   
#Medium level functions#
########################

def orientation_angle(im, approxangle=None,* ,isshiftdft=False,truesize=None,
                      rotateAngle=None):
    """Give the highest contribution to the orientation
    
    Parameters
    ----------
    im: 2d array
        The image
    approxangle: number, optional
        The approximate angle (None if unknown)
    isshiftdft: Boolean, default False
        True if the image has been processed (DFT, fftshift)
    truesize: 2 numbers, optional
        Truesize of the image if isshiftdft is True
    rotateAngle: number, optional
        The diagonals are more sensitives than the axis. 
        rotate the image to avoid pixel orientation (flat or diagonal)
    
    Returns
    -------
    angle: number
        The orientation of the image
    
    Notes
    -----
    if approxangle is specified, search only within +- pi/4    
    """
    im=np.asarray(im)
    
    #If we rotate the image first
    if rotateAngle is not None and not isshiftdft:   
        #compute the scale corresponding to the image
        scale=np.sqrt(.5*(1+(np.tan(rotateAngle)-1)**2/
                            (np.tan(rotateAngle)+1)**2))
        #rotate the image
        im=rotate_scale(im,rotateAngle,scale)
    
    
    
    #compute log fft (nearest interpolation as line go between pixels)
    lp=polar_fft(im,isshiftdft=isshiftdft,
                            logoutput=False, interpolation='nearest',
                            truesize=truesize)  
    #get distribution
    adis=lp.sum(-1)
    if approxangle is not None:
        #-np.pi/2 as we are in fft. +- pi/4 is search window
        amin=clamp_angle(approxangle-np.pi/4-np.pi/2)
        amax=clamp_angle(approxangle+np.pi/4-np.pi/2)
        angles=np.linspace(-np.pi/2,np.pi/2,lp.shape[0],endpoint=False) 
        if amin>amax:
            adis[np.logical_and(angles > amax, angles < amin)]=adis.min()
        else:
            adis[np.logical_or(angles > amax, angles < amin)]=adis.min()
            
    #get peak pos
    ret=get_peak_pos(adis,wrap=True)
    anglestep = np.pi/lp.shape[0]
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(anglestep*np.arange(len(adis)),adis)
    #"""
    
    #return max (not -pi/2 as 0 is along y and we want 0 alonx x.)
    #the transformation is basically "For Free"
    ret = clamp_angle(ret*anglestep)
    
    #Substract rotateAngle to get the original image
    if rotateAngle is not None:
        ret=clamp_angle(ret-rotateAngle)
        
        
    return ret

def orientation_angle_2(im, nangle=None,isshiftdft=False,rotateAngle=None):
    """Give the highest contribution to the orientation
    
    Parameters
    ----------
    im: 2d array
        The image
    nangle: number, optional
        The number of angles checked for
    isshiftdft: Boolean, default False
        True if the image has been processed (DFT, fftshift)
    rotateAngle: number, optional
        The diagonals are more sensitives than the axis. 
        rotate the image to avoid pixel orientation (flat or diagonal)
    
    Returns
    -------
    angle: number
        The orientation of the image
      
    """
    im=np.asarray(im, dtype=np.float32)
    
    #If we rotate the image first
    if rotateAngle is not None and not isshiftdft:   
        #compute the scale corresponding to the image
        scale=np.sqrt(.5*(1+(np.tan(rotateAngle)-1)**2/
                            (np.tan(rotateAngle)+1)**2))
        #rotate the image
        im=rotate_scale(im,rotateAngle,scale)
    else:
        rotateAngle=0
        
    #get dft if not already done
    if not isshiftdft:
        im=centered_mag_sq_ccs(dft_optsize(im))
    
    #the center is shifted from 0,0 to the ~ center 
    #(eg. if 4x4 image, the center is at [2,2], as if 5x5)
    qshape = np.asarray([im.shape[0]//2,im.shape[1]])
    center = np.asarray([qshape[0],0]) 
    im[qshape[0],0]=0
    
    #if the angle Step is not given, take the number of pixel
    #on the perimeter as the target #=range/step
    
    if nangle is None:
        nangle=np.min(im.shape)#range is pi, nbangle = 2r =~pi r
        
    #get the theta range
    theta=np.linspace(-np.pi/2,np.pi/2,nangle,endpoint=False, dtype=np.float32)
#    X=np.arange(im.shape[1])-center[1]
#    Y=-(np.arange(im.shape[0])-center[0])
#    YG, XG =np.meshgrid(Y,X,indexing='ij')   
    
    YG,XG=np.ogrid[-center[0]:im.shape[0]-center[0],
                       -center[1]:im.shape[1]-center[1]]
    YG=-YG
    values=np.empty(theta.shape)
    
    #save useful value for computations
    ylim=center[0]+1
    
    
    
    #The negative upper limit is the top right corner
    upper=np.arctan((YG+.5)/(XG+.5))
    #save lower for later
    lower=upper.copy()
    #The positive is top left
    upper[:ylim,1:]=upper[:ylim,:-1]
    #for X=0, the top left is meaningless (atan gives only up to pi/2)
    #should be pi/2 but not important as angle are smaller anyway
    #Therefore avoid pi/2<pi/2 being false
    upper[:ylim,0]=np.pi
    
    #idem for lower limit: lower right for positives
    lower[:-1,:]=lower[1:,:]
    lower[-1,:]=np.arctan((YG[-1,:]-.5)/(XG[-1,:]+.5))
    #For negative t, the lower left
    lower[ylim:,1:]=lower[ylim:,:-1]
    #correct arctan
    lower[ylim:,0]=-np.pi
    
    for i,t in enumerate(theta.flat):
        cond=np.logical_and(t>lower, 
                            t<upper)
        values[i]=np.sum(im[cond])      
    
    """
    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.imshow(np.log(im[qshape[0]-5:qshape[0]+5,:5]),interpolation='none')
    plt.figure()
    plt.plot(theta, values,'x')
#    plt.plot(ir.clamp_angle(0.0555-np.pi/2-rotateAngle)*np.array([1,1]),[np.amin(values),np.amax(values)])
    #"""
    
    return clamp_angle(theta[np.argmax(values)]+np.pi/2+rotateAngle)
    
def dft_optsize(im, shape=None):
    """Resize image for optimal DFT and computes it
    
    Parameters
    ----------
    im: 2d array
        The image
    shape: 2 numbers, optional
        The shape of the output image (None will optimize the shape)
    
    Returns
    -------
    dft: 2d array
        The dft in CCS representation
    
    Notes
    -----
    Th shape shoulb be a product of 2, 3, and 5
    """
    im=np.asarray(im)
    #save shape
    initshape=im.shape
    #get optimal size
    if shape is None:
        ys=cv2.getOptimalDFTSize(initshape[0]) 
        xs=cv2.getOptimalDFTSize(initshape[1])
        shape=[ys,xs]
    #Add zeros to go to optimal size
    im = cv2.copyMakeBorder(im, 0, shape[0] - initshape[0],
                                0, shape[1] - initshape[1], 
                                borderType=cv2.BORDER_CONSTANT, value=0);
    #Compute dft ignoring 0 rows (0 columns can not be optimized)
    f = cv2.dft(im,nonzeroRows=initshape[0])
    return f
    
    
def dft_optsize_same(im0,im1):
    """Resize 2 image same size for optimal DFT and computes it
    
    Parameters
    ----------
    im0: 2d array
        First image
    im1: 2d array
        Second image
    
    Returns
    -------
    dft0: 2d array
        The dft of the first image 
    dft1: 2d array
        The dft of the second image
        
    Notes
    -----
    dft0 and dft1 will have the same size
    """
    im0=np.asarray(im0)
    im1=np.asarray(im1)
    #save shape
    shape0=im0.shape
    shape1=im1.shape
    #get optimal size
    ys=max(cv2.getOptimalDFTSize(shape0[0]),
           cv2.getOptimalDFTSize(shape1[0]))
    xs=max(cv2.getOptimalDFTSize(shape0[1]),
           cv2.getOptimalDFTSize(shape1[1]))
    shape=[ys,xs]
    f0=dft_optsize(im0,shape=shape)
    f1=dft_optsize(im1,shape=shape)
    return f0,f1
    
def rotate_scale(im, angle, scale, borderValue=0, interp=cv2.INTER_CUBIC):
    """Rotates and scales the image
    
    Parameters
    ----------
    im: 2d array
        The image
    angle: number
        The angle, in radians, to rotate
    scale: positive number
        The scale factor
    borderValue: number, default 0
        The value for the pixels outside the border (default 0)
    
    Returns
    -------
    im: 2d array
        the rotated and scaled image
    
    Notes
    -----
    The output image has the same size as the input. 
    Therefore the image may be cropped in the process.
    """
    im=np.asarray(im,dtype=np.float32)
    rows,cols = im.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle*180/np.pi,1/scale)
    im = cv2.warpAffine(im,M,(cols,rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        flags=interp,
                        borderValue=borderValue)#REPLICATE
    return im
    
def shift_image(im, shift, borderValue=0):
    """shift the image
    
    Parameters
    ----------
    im: 2d array
        The image
    shift: 2 numbers
        (y,x) the shift in y and x direction
    borderValue: number, default 0
        The value for the pixels outside the border (default 0)
        
    Returns
    -------
    im: 2d array
        The shifted image
    
    Notes
    -----
    The output image has the same size as the input. 
    Therefore the image will be cropped in the process.
    """
    im=np.asarray(im,dtype=np.float32)
    rows,cols = im.shape
    M = np.asarray([[1,0,shift[1]],[0,1,shift[0]]],dtype=np.float32)
    return cv2.warpAffine(im,M,(cols,rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        flags=cv2.INTER_CUBIC,
                        borderValue=borderValue)
    
def rotate_scale_shift(im, angle, scale, shift, borderValue=0):
    """Rotates and scales the image
    
    Parameters
    ----------
    im: 2d array
        The image
    angle: number
        The angle, in radians, to rotate
    scale: positive number
        The scale factor
    shift: 2 numbers
        (y,x) the shift in y and x direction
    borderValue: number, default 0
        The value for the pixels outside the border (default 0)
    
    Returns
    -------
    im: 2d array
        the rotated, scaled, and shifted image
    
    Notes
    -----
    The output image has the same size as the input. 
    Therefore the image may be cropped in the process.
    """
    im=np.asarray(im,dtype=np.float32)
    rows,cols = im.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle*180/np.pi,1/scale)
    M[0,2]+=shift[1]
    M[1,2]+=shift[0]
    im = cv2.warpAffine(im,M,(cols,rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        flags=cv2.INTER_CUBIC,
                        borderValue=borderValue)#REPLICATE
    return im
    
    
def polar_fft(im, nangle=None, radiimax=None, *, isshiftdft=False,
              truesize=None, logpolar=False, logoutput=False, 
              interpolation='bilinear'):
    """Return dft in polar (or log-polar) units, the angle step 
    (and the log base)
    
    Parameters
    ----------
    im: 2d array
        The image
    nangle: number, optional
        The number of angles in the polar representation
    radiimax: number, optional
        The number of radius in the polar representation
    isshiftdft: boolean, default False
        True if the image is pre processed (DFT + fftshift)
    truesize: 2 numbers, required if isshiftdft is True
        The true size of the image
    logpolar: boolean, default False
        True if want the log polar representation instead of polar
    logoutput: boolean, default False
        True if want the log of the output
    interpolation: string, default 'bilinear'
        ('bicubic', 'bilinear', 'nearest') The interpolation 
        technique. (For now, avoid bicubic)
    
    Returns
    -------
    im: 2d array
        The (log) polar representation of the input image
    log_base: number, only if logpolar is True
        the log base if this is log polar representation
    
    Notes
    -----
    radiimax is the maximal radius (log of radius if logpolar is true).
    if not provided, it is deduced from the image size
    
    To get log-polar, set logpolar to True
    log_base is the base of the log. It is deduced from radiimax.
    Two images that will be compared should therefore have the same radiimax.
    """
    im=np.asarray(im, dtype=np.float32)
    #get dft if not already done
    if not isshiftdft:
        truesize=im.shape
        #substract mean to avoid having large central value
        im=im-im.mean()
        im=centered_mag_sq_ccs(dft_optsize(im))
    #We need truesize! otherwise border effects.
    assert(truesize is not None)
    
    #the center is shifted from 0,0 to the ~ center 
    #(eg. if 4x4 image, the center is at [2,2], as if 5x5)
    qshape = np.asarray([im.shape[0]//2,im.shape[1]])
    center = np.asarray([qshape[0],0]) 
    
    #if the angle Step is not given, take the number of pixel
    #on the perimeter as the target #=range/step
    
    if nangle is None:
        #TODO: understand why nangle need to be exactly truesize
        nangle=np.min(truesize)#range is pi, nbangle = 2r =~pi r
#        nangle-=2
        
    #get the theta range
    theta=np.linspace(-np.pi/2,np.pi/2,nangle,endpoint=False, dtype=np.float32)
    
    #For the radii, the units are comparable if the log_base and radiimax are
    #the same. Therefore, log_base is deduced from radiimax
    #The step is assumed to be 1
    if radiimax is None:
        radiimax = qshape.min()
    
    #also as the circle is an ellipse in the image, 
    #we want the radius to be from 0 to 1
    if logpolar:
        #The log base solves log_radii_max=log_{log_base}(linear_radii_max)
        #where we decided arbitrarely that linear_radii_max=log_radii_max
        log_base=np.exp(np.log(radiimax) / radiimax)
        radius = ((log_base ** np.arange(0,radiimax,dtype=np.float32))
                    /radiimax)
    else:
        radius = np.linspace(0,1,radiimax, endpoint=False,
                                    dtype=np.float32)
    
    #get x y coordinates matrix (The units are 0 to 1, therefore a circle is
    #represented as an ellipse)
    y=cv2.gemm(np.sin(theta),radius,qshape[0],0,0,
               flags=cv2.GEMM_2_T)+center[0]
    x=cv2.gemm(np.cos(theta),radius,qshape[1],0,0,
               flags=cv2.GEMM_2_T)+center[1]
    interp=cv2.INTER_LINEAR
    if interpolation == 'bicubic':
        interp=cv2.INTER_CUBIC
    if interpolation == 'nearest':
        interp=cv2.INTER_NEAREST
    
    #get output
    output=cv2.remap(im,x,y, interp)#LINEAR, CUBIC,LANCZOS4
    #apply log
    if logoutput:
        output=cv2.log(output)
    
    if logpolar:
        return output, log_base
    else:
        return output
        
        
##################    
#Helper functions#
##################
    
    
def pad_img(im,pad):
    """Pad positively with 0 or negatively (cut)
    
    Parameters
    ----------
    im: 2d array
        The image
    pad: 4 numbers
        (ytop, ybottom, xleft, xright) or (imin, imax, jmin, jmax)
        
    Returns
    -------
    im: 2d array
        The padded (or cropped) image
    offset: 2 numbers
        The offset related to the input image
        
    Notes
    -----
    This changes the size of the image
    """
    im=np.asarray(im)
    pad=np.asarray(pad)
    #get shape
    shape=im.shape
    #extract offset from padding 
    offset=-pad[::2]
    #if the padding is negatif, cut the matrix
    cut=pad<0
    if cut.any():
        #Extract value for pad
        cut*=pad
        #the left/top components should be positive
        cut[::2]*=-1
        #The right/bottom components can't be 0, replace by shape0
        cut[1::2]+=(cut[1::2]==0)*shape
        #cut the image
        im=im[cut[0]:cut[1],cut[2]:cut[3]]
    #extract positive padding
    ppad=pad>0
    if ppad.any():
        pad=pad*ppad
        #separate pad for application on matrix
        ypad=(pad[0],pad[1])
        xpad=(pad[2],pad[3])
        #prepare matrix
        im=np.lib.pad(im,(ypad,xpad),mode='mean')
    return im, offset
    
    
def clamp_angle(angle):
    """return a between -pi/2 and pi/2 (in fourrier plane, +pi is the same)
    
    Parameters
    ----------
    angle: number
        The angle to be clamped
        
    Returns
    -------
    angle: number
        The clamped angle
    
    """
    return (angle+np.pi/2)%np.pi - np.pi/2
    
    
def ccs_normalize(compIM,ccsnorm):
    """ normalize the ccs representation 
    
    Parameters
    ----------
    compIM: 2d array
        The CCS image in CCS representation
    ccsnorm: 2d array
        The normalization matrix in ccs representation
    
    Returns
    -------
    compIM: 2d array
        The normalized CCS image
    
    Notes
    -----
    (basically an element wise division for CCS)
    Should probably not be used from outside
    """
    compIM=np.asarray(compIM)
    ccsnorm=np.asarray(ccsnorm)
    ys=ccsnorm.shape[0]
    xs=ccsnorm.shape[1]
    #start with first column
    ccsnorm[2::2,0]=ccsnorm[1:ys-1:2,0]
    #continue with middle columns
    ccsnorm[:,2::2]=ccsnorm[:,1:xs-1:2]
    #finish whith last row if even
    if xs%2 is 0:
        ccsnorm[2::2,xs-1]=ccsnorm[1:ys-1:2,xs-1]
    res= compIM/ccsnorm  
    #solve problem with 0/0
    res[compIM==0]=0
    return res
      
      
def gauss_fit(X,Y):
    """
    Fit the function to a gaussian.
    
    Parameters
    ----------
    X: 1d array
        X values
    Y: 1d array
        Y values
        
    Returns
    -------
    (The return from scipy.optimize.curve_fit)
    popt : array
        Optimal values for the parameters
    pcov : 2d array
        The estimated covariance of popt.
    
    Notes
    -----
    /!\ This uses a slow curve_fit function! do not use if need speed!
    """
    X=np.asarray(X)
    Y=np.asarray(Y)
    #Can not have negative values
    Y[Y<0]=0
    #define gauss function
    def gauss(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    #get first estimation for parameter
    mean=(X*Y).sum()/Y.sum()
    sigma=np.sqrt((Y*((X-mean)**2)).sum()/Y.sum())
    height=Y.max()
    #fit with curve_fit
    return curve_fit(gauss,X,Y,p0=[height,mean,sigma])
    
    
def gauss_fit_log(X,Y):
    """
    Fit the log of the input to the log of a gaussian.
    
    Parameters
    ----------
    X: 1d array
        X values
    Y: 1d array
        Y values
        
    Returns
    -------
    mean: number
        The mean of the gaussian curve
    var: number
        The variance of the gaussian curve
            
    Notes
    -----
    The least square method is used. 
    As this is a log, make sure the amplitude is >> noise
    
    See the gausslog_sympy.py file for explaination
    """
    X=np.asarray(X)
    Y=np.asarray(Y)
    #take log data
    Data=np.log(Y)
    #Get Di and Xi
    D=[(Data*X**i).sum() for i in range(3)]
    X=[(X**i).sum() for i in range(5)]
    #compute numerator and denominator for mean and variance
    num=(D[0]*(X[1]*X[4] - X[2]*X[3]) +
         D[1]*(X[2]**2   - X[0]*X[4]) +
         D[2]*(X[0]*X[3] - X[1]*X[2]))
    den=2*(D[0]*(X[1]*X[3] - X[2]**2) + 
           D[1]*(X[1]*X[2] - X[0]*X[3]) +
           D[2]*(X[0]*X[2] - X[1]**2))
    varnum=(-X[0]*X[2]*X[4] + X[0]*X[3]**2 + X[1]**2*X[4] -
            2*X[1]*X[2]*X[3] + X[2]**3)
    #if denominator is 0, can't do anything
    if abs(den) < 0.00001:
#        print('Warning: zero denominator!',den)
        return np.nan,np.nan
    #compute mean and variance
    mean=num/den
    var=varnum/den
    #if variance is negative, the data are not a gaussian
    if var<0:
#        print('Warning: negative Variance!',var)
        return np.nan,np.nan
    return mean,var
    
    
def center_of_mass(X,Y):
    """Get center of mass
    
    Parameters
    ----------
    X: 1d array
        X values
    Y: 1d array
        Y values
        
    Returns
    -------
    res: number
        The position of the center of mass in X
        
    Notes
    -----
    Uses least squares
    """
    X=np.asarray(X)
    Y=np.asarray(Y)
    return (X*Y).sum()/Y.sum()
    
    
def get_peak_pos(im,wrap=False):
    """Get the peak position with subpixel precision
    
    Parameters
    ----------
    im: 2d array
        The image containing a peak
    wrap: boolean, defaults False
        True if the image reoresents a torric world
        
    Returns
    -------
    [y,x]: 2 numbers
        The position of the highest peak with subpixel precision
        
    Notes
    -----
    This is a bit hacky and could be improved
    """ 
    im=np.asarray(im)
    #remove invalid values (assuming im>0)
    im[np.logical_not(np.isfinite(im))] = 0
    #remove mean
    im=im-im.mean()
    #get maximum value
    argmax=im.argmax()
    dsize=im.size
    #get cut value (30% biggest peak)
    #TODO: choose less random value
    cut = .3*im[argmax]
    #isolate peak
    peak=im>cut
    peak,__=label(peak)
    #wrap border
    if wrap and peak[0]!=0 and peak[-1]!=0 and peak[0]!=peak[-1]:
        peak[peak==peak[-1]]=peak[0]
    #extract peak
    peak=peak==peak[argmax]
    
    #get values along X and Y
    X=np.arange(dsize)[peak]
    Y=im[peak]
     #wrap border
    if wrap:
        #wrap X values d
        X[X>dsize//2]-=dsize
        
    #remove argmax as in X**4 X should be small
    offset=X[Y==Y.max()][0]
    X-=offset
    
    #We want to fit in a radius of 3 around the center
    Y=Y[abs(X)<3]
    X=X[abs(X)<3]
    
    #if>2, use fit_log
    if peak.sum() > 2:
        ret,__=gauss_fit_log(X,Y)
        #if fails, use center_of_mass
        if ret is np.nan:
            ret=center_of_mass(X,Y)
    elif peak.sum() >1:
        #If only 2 pixel, gauss fitting is imposible, use center_of_mass
        ret=center_of_mass(X,Y)
    else:
        #1 px peak is easy
        ret=X[0]
        
        
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(X,Y,'x',label='im')
    plt.plot([ret,ret],[1,Y.max()],label='logfit')
    plt.plot([X.min(),X.max()],[cut,cut])
    plt.plot([X.min(),X.max()],[im.std(),im.std()])
    #"""
    
    return ret+offset
    

def get_extent(origin, shape):
    """Computes the extent for imshow() (see matplotlib doc)
    
    Parameters
    ----------
    origin: 2 numbers
        The origin of the second image in the first image coordiantes
    shape: 2 numbers
        The shape of the image
        
    Returns
    -------
    extent: 2 numbers
        The extent
    """
    return [origin[1],origin[1]+shape[1],origin[0]+shape[0],origin[0]]
      
def centered_mag_sq_ccs(im):
    """return centered squared magnitude
    
    Parameters
    ----------
    im: 2d array
        A CCS DFT image
    Returns
    -------
    im: 2d array
        A centered image of the magnitude of the DFT
        
    Notes
    -----
    Check doc Intel* Image Processing Library
    https://www.comp.nus.edu.sg/~cs4243/doc/ipl.pdf
    
    The center is at position (ys//2, 0)    
    """
    im=np.asarray(im)
    #multiply image by image* to get squared magnitude
    im=cv2.mulSpectrums(im,im,flags=0,conjB=True)    
    
    ys=im.shape[0]
    xs=im.shape[1]
    
    #get correct size return
    ret=np.zeros((ys,xs//2+1))
    
    #first column:
    #center
    ret[ys//2,0]=im[0,0]
    #bottom
    ret[ys//2+1:,0]=im[1:ys-1:2,0]   
    #top (Inverted copy bottom)
    ret[ys//2-1::-1,0]=im[1::2,0] 
    
    #center columns
    ret[ys//2:,1:]=im[:(ys-1)//2+1,1::2]
    ret[:ys//2,1:]=im[(ys-1)//2+1:,1::2]
    
    #correct last line if even
    if xs%2 is 0:
        ret[ys//2+1:,xs//2]=im[1:ys-1:2,xs-1]   
        ret[:ys//2,xs//2]=0
        
    return ret

def is_overexposed(im):
    """Simple test to check if image is overexposed
    
    Parameters
    ----------
    im: 2d array integer
        the image
    Returns
    -------
    overexposed: Bool
        Is the image overexposed  
    """
    diffbincount=np.diff(np.bincount(np.ravel(im)))
    overexposed=diffbincount[-1]>np.std(diffbincount)
    return overexposed