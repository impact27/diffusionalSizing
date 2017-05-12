"""
Created on Wed Apr  5 16:58:39 2017

@author: quentinpeter
"""
import sys
sys.path.append('lib/')
import diffusion_device.profile as dp
import diffusion_device.channel_image as dd12
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot, figure
from glob import glob
import re
from natsort import natsorted

#==============================================================================
# Settings
#==============================================================================

#File Name. If using UV, [image filename, background filename]
bgfn='Data/20160905_BSA_low_con_sizing/6uM_BSA_25mM_phosphate_pH_8_23C/bg_500ms/*.tif'
imfn='Data/20160905_BSA_low_con_sizing/15uM_BSA_25mM_phosphate_pH_8_350ulh_23C/im_*.tif'
filename=[natsorted(glob(imfn)),
          natsorted(glob(bgfn))]
filename='Data/Tom/60uM BSA+600uM OPA 80ul-h 1/*.tif'
filename=natsorted(glob(filename))[:13]

#Settings
Wz=53*1e-6 #m Height of the device
Wy=300*1e-6 #m Width of the device
rmin=.5e-9#m test radii
rmax=10e-9#m test radii
rstep=.1e-9#m test radii
ActualFlowRate=None#350/2 #ulph. If None, guess from filename (looks for ulph)
save=False
pixsize=.84e-6#pixel sixe in m

#Advanced settings
ignore=10e-6 #m Distance from the sides to ignore
initmode='none' #use that to filter the first position
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
readingpos=dd12.defaultReading12Pos()
data_dict={}
if ActualFlowRate is None:
    fn=filename
    if len(np.shape(fn))>1: 
        fn=fn[0]
    ActualFlowRate=int(re.findall('.*?([\d\.]+)ul?-?p?h.*',fn[0])[0])

# Get radius and LSE
radius=dd12.size_images(filename,ActualFlowRate,Wz,pixsize,readingpos,chanWidth=Wy
     ,Rs=test_radii,data_dict=data_dict,ignore=ignore,initmode=initmode)

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
plt.title('r= {:.2f} nm, LSE = {:.2e}'.format(
        radius*1e9, lse))   
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
Pixel size:\t {} um""".format(filename,
            Wz*1e6,
            Wy*1e6,
            rmin*1e9,
            rmax*1e9,
            rstep*1e9,
            ActualFlowRate,
            ignore*1e6,
            pixsize*1e6))
        if initmode != 'none':
            f.write("Initial profile processing:\t {}\n".format(initmode))
            
    with open(name+'_result.txt','wb') as f:
        f.write("""Radius: {:f} nm
LSE: {:e}
Profiles:
""".format(radius*1e9,lse).encode())
        np.savetxt(f,profiles)
        f.write('Fits:\n'.encode())
        np.savetxt(f,fits)
        f.write('Test radii:\n'.encode())
        np.savetxt(f,test_radii)
        f.write('Reading Position:\n'.encode())
        np.savetxt(f,readingpos)
        
