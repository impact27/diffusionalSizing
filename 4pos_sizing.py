"""
Created on Wed Apr  5 16:58:39 2017

@author: quentinpeter
"""
import sys
sys.path.append('lib/')
import diffusion_device.profile as dp
import diffusion_device.four_channels_image as dd4
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
import re

#==============================================================================
# Settings
#==============================================================================

#File Name. If using UV, [image filename, background filename]
bgfn='sampleData/UVbg.tif'
imfn='sampleData/UVim300ulph.tif'
filename=[imfn,bgfn]
#filename='sampleData/Brightim900ulph.tif'

#Settings
Wz=53*1e-6 #m Height of the device
Wy=100*1e-6 #m Width of the device
rmin=.5e-9#m test radii
rmax=10e-9#m test radii
rstep=.1e-9#m test radii
ActualFlowRate=None #ulph. If None, guess from filename (looks for ulph)
save=False

#Advanced settings
readstartpos=1000e-6 #m Distance from the center of the image to the first edge after the nozzle
ignore=5e-6 #m Distance from the sides to ignore
initmode='none' #use that to filter the first position
fit_position_number=None #The positions to take into account. None means all
imSlice=None #The image slice to analyse. None mean all, otherwise slice(min,max)

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
    ActualFlowRate=int(re.findall('.*?([\d\.]+)ul?-?p?h.*',fn)[0])

# Get radius and LSE
radius=dd4.size_image(filename,ActualFlowRate,Wz,Wy,readingpos
     ,test_radii,data_dict=data_dict,ignore=ignore,initmode=initmode, 
     fit_position_number=fit_position_number, imSlice=imSlice)

assert 'profiles' in data_dict, 'Profiles not found'
lse=np.sqrt(np.mean(np.square(data_dict['profiles'][1:]-data_dict['fits'])))

#Get profiles and fit
profiles=data_dict['profiles']
fits=[data_dict['initprof'],*data_dict['fits']]
pixel_size=data_dict['pixsize']

#==============================================================================
# Plot
#==============================================================================

X=np.arange(len(dp.get_fax(profiles)))*pixel_size*1e6
figure()
plot(X,dp.get_fax(profiles))
plot(X,dp.get_fax(fits))
plt.title('r= {:.2f} nm, LSE = {:.2e}, pixel = {:.3f} um'.format(
        radius*1e9, lse, pixel_size*1e6))   
plt.xlabel('Position [$\mu$m]')
plt.ylabel('Normalised amplitude')

#==============================================================================
# Save
#==============================================================================

if save:
    from os import path
    fn=filename
    if len(np.shape(fn))!=0:
        fn=fn[0]
    name=path.splitext(fn)[0]
    plt.savefig(name+'_fig.pdf')
    with open(name+'_settings.txt','w') as f:
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
            
    with open(name+'_result.txt','wb') as f:
        f.write("""Radius: {:f} nm
LSE: {:e}
Apparent pixel size: {:f} um
Profiles:
""".format(radius*1e9,lse,pixel_size*1e6).encode())
        np.savetxt(f,profiles)
        f.write('Fits:\n'.encode())
        np.savetxt(f,fits)
        f.write('Test radii:\n'.encode())
        np.savetxt(f,test_radii)
        f.write('Reading Position:\n'.encode())
        np.savetxt(f,readingpos)
        
