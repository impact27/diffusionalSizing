# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:47:11 2017

@author: quentinpeter
"""
import sys
sys.path.append('lib/')
import diffusion_device.profile as dp
import diffusion_device.four_channels_image as dd4
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
import tifffile
import re
from os import path
from image_registration.image import is_overexposed

#==============================================================================
# Settings
#==============================================================================

#File Name.
filename='Film/327.68ul-h-50um device.tif'
filename='Film/389ulph-64.5um device.tif'

#Settings
Wz=53*1e-6 #m Height of the device
Wy=100*1e-6 #m Width of the device
rmin=.5e-9#m test radii
rmax=10e-9#m test radii
rstep=.1e-9#m test radii
ActualFlowRate=None #ulph. If None, guess from filename (looks for ulph)
plotpos=[0,5,10]#Frames to plot
save=True

#Advanced settings
readstartpos=300e-6 #m Distance from the center of the image to the first edge after the nozzle
ignore=5e-6 #m Distance from the sides to ignore
initmode='none' #use that to filter the first position
fit_position_number=None #The positions to take into account. None means all
imSlice=slice(150,-1)#None #The image slice to analyse. None mean all, otherwise slice(min,max)
framesSlice=slice(0,250)
flatten=True# Only for bright field images
#==============================================================================
# Fit----------DO NOT CHANGE BELOW THIS POINT----------------------------------
#==============================================================================
#        _  _
#  ___ (~ )( ~)
# /   \_\ \/ /
#|   D_ ]\ \/
#|   D _]/\ \
# \___/ / /\ \
#      (_ )( _)

# Infer variables
test_radii=np.arange(rmin,rmax,rstep) 
readingpos=dd4.defaultReadingPos(readstartpos)
data_dict={}
if ActualFlowRate is None:
    fn=filename
    if len(np.shape(fn))!=0: 
        fn=fn[0]
    ActualFlowRate=float(re.findall('.*?([\d\.]+)ul?-?p?h.*',fn)[0])


ims=tifffile.imread(filename)

if framesSlice is not None:
    ims=ims[framesSlice]

def size(im,ActualFlowRate,Wz,Wy,readingpos,Rs,ignore=ignore,
         fit_position_number=fit_position_number,
         imSlice=imSlice,initmode=initmode,flatten=flatten):
    data_dict={}
    r= dd4.size_image(
        im,ActualFlowRate,Wz,Wy,readingpos,Rs,ignore=ignore,
        data_dict=data_dict,initmode=initmode,
        fit_position_number=fit_position_number,imSlice=imSlice,
        flatten=flatten)
    return r,data_dict
    

overexposed=[is_overexposed(im) for im in ims]

results=[size(im,ActualFlowRate=ActualFlowRate,Wz=Wz,Wy=Wy,
       readingpos=readingpos,Rs=test_radii,ignore=ignore,imSlice=imSlice,
       fit_position_number=fit_position_number) for im in ims]
    

#%%
radii=np.zeros(len(results))
LSE=np.zeros(len(results))*np.nan
pixs=np.zeros(len(results))*np.nan
for i,rval in enumerate(results):
    radii[i]=rval[0]
    if 'fits' in rval[1]:
        ps=rval[1]['pixsize']
        pixs[i]=ps
        ignorepix=int(ignore/ps)
        LSE[i]=np.sqrt(np.mean(np.square(
                rval[1]['profiles'][1:,ignorepix:-ignorepix]
                -rval[1]['fits'][:,ignorepix:-ignorepix])))
        

fn=filename
if len(np.shape(fn))!=0:
    fn=fn[0]
base_name=path.splitext(fn)[0]

#%%
x=np.arange(len(radii))
valid=np.logical_not(overexposed)
figure()
plot(x[valid],radii[valid]*1e9,'x',label='data')
plt.xlabel('Frame number')
plt.ylabel('Radius [nm]')
if np.any(overexposed):
    plot(x[overexposed],radii[overexposed]*1e9,'x',label='overexposed data')
    plt.legend()
if save:
    plt.savefig(base_name+'_R_fig.pdf')
    
figure()
plot(x[valid],LSE[valid],'x',label='regular')
plt.xlabel('Frame number')
plt.ylabel('Least square error')
if np.any(overexposed):
    plot(x[overexposed],LSE[overexposed],'x',label='overexposed')
    plt.legend()
if save:
    plt.savefig(base_name+'_LSE_fig.pdf')
    
figure()
plot(pixs*1e6,'x')
plt.xlabel('Frame number')
plt.ylabel('Pixel size')
if save:
    plt.savefig(base_name+'_pixel_size_fig.pdf')

if save:
    with open(base_name+'_settings.txt','w') as f:
        f.write("""File name:\t {}
Channel height:\t {} um
Channel width:\t {} um
Test radii:\t {} to {} nm , step {} nm
Flow rate:\t {} ulph
Border ignore: \t {} um
First reading position:\t {} um""".format(filename,
            Wz*1e6,
            Wy*1e6,
            rmin*1e9,
            rmax*1e9,
            rstep*1e9,
            ActualFlowRate,
            ignore*1e6,
            readstartpos*1e6))
        if initmode != 'none':
            f.write("Initial profile processing:\t {}\n".format(initmode))
        if fit_position_number is not None:
            f.write("Positions fitted:\t {}\n".format(fit_position_number))
        if imSlice is not None:
            f.write("Image slice:\t {}\n".format(imSlice))
            
    with open(base_name+'_result.txt','wb') as f:
        f.write('Radii:\n'.encode())
        np.savetxt(f,radii)
        f.write('Least square error:\n'.encode())
        np.savetxt(f,LSE)
        if np.any(overexposed):
            f.write('Overexposed Frames:\n'.encode())
            np.savetxt(f,overexposed)
        f.write('Pixel size:\n'.encode())
        np.savetxt(f,pixs)
        f.write('Test radii:\n'.encode())
        np.savetxt(f,test_radii)
        f.write('Reading Position:\n'.encode())
        np.savetxt(f,readingpos)
        

        
#%%
plotpos=np.asarray(plotpos)

for pos in plotpos[plotpos<len(results)]:
    if 'profiles' in results[pos][1]:
        profs=dp.get_fax(results[pos][1]['profiles'])
        X=np.arange(len(profs))*pixs[pos]*1e6
        figure()
        plot(X,profs)
        
        if 'initprof' in results[pos][1] and 'fits' in results[pos][1]:
            fits=dp.get_fax([results[pos][1]['initprof'],
                            *results[pos][1]['fits']])
        
            plot(X,fits)
        plt.title('r= {:.2f} nm, LSE = {:.2e}, pixel = {:.3f} um'.format(
                radii[pos]*1e9, LSE[pos], pixs[pos]*1e6))   
        plt.xlabel('Position [$\mu$m]')
        plt.ylabel('Normalised amplitude')
