# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:32:10 2017

@author: quentinpeter
"""
import numpy as np
from scipy.linalg import toeplitz

#%%

    
    
def poiseuille(Zgrid,Ygrid,Wz,Wy,Q,get_interface=False):
    """
    Compute the poiseuille flow profile
    
    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Wz: float
        Channel height [m]
    Wy: float 
        Channel width [m]
    Q:  float
        The flux in the channel in [ul/h]
    get_interface: Bool, defaults False
        Also returns poisuille flow between pixels
    Returns
    -------
    V: 2d array
        The poiseuille flow
    if get_interface is True:
    Viy: 2d array
        The poiseuille flow between y pixels
    Viz: 2d array
        The poiseuille flow between z pixels
    """
        
    #Poiseuille flow
    V=np.zeros((Zgrid,Ygrid),dtype='float64')    
    for j in range(Ygrid):
        for i in range(Zgrid):
            nz=np.arange(1,100,2)[:,None]
            ny=np.arange(1,100,2)[None,:]
            V[i,j]=np.sum(1/(nz*ny*(nz**2/Wz**2+ny**2/Wy**2))*
                      (np.sin(nz*np.pi*(i+.5)/Zgrid)*
                       np.sin(ny*np.pi*(j+.5)/Ygrid)))
    Q/=3600*1e9 #transorm in m^3/s
    #Normalize
    normfactor=Q/(np.mean(V)* Wy *Wz)
    V*=normfactor
    
    if not get_interface:
        return V
    #Y interface
    Viy=np.zeros((Zgrid,Ygrid-1),dtype='float64')    
    for j in range(1,Ygrid):
        for i in range(Zgrid):
            nz=np.arange(1,100,2)[:,None]
            ny=np.arange(1,100,2)[None,:]
            Viy[i,j-1]=np.sum(1/(nz*ny*(nz**2/Wz**2+ny**2/Wy**2))*
                      (np.sin(nz*np.pi*(i+.5)/Zgrid)*
                       np.sin(ny*np.pi*(j)/Ygrid)))
    Viy*=normfactor
    #Z interface       
    Viz=np.zeros((Zgrid-1,Ygrid),dtype='float64')    
    for j in range(Ygrid):
        for i in range(1,Zgrid):
            nz=np.arange(1,100,2)[:,None]
            ny=np.arange(1,100,2)[None,:]
            Viz[i-1,j]=np.sum(1/(nz*ny*(nz**2/Wz**2+ny**2/Wy**2))*
                      (np.sin(nz*np.pi*(i)/Zgrid)*
                       np.sin(ny*np.pi*(j+.5)/Ygrid)))
            
    Viz*=normfactor
    return V,Viy,Viz
#%%    
def stepMatrix(Zgrid,Ygrid,Wz,Wy,Q,outV=None):
    """
    Compute the step matrix and corresponding position step
    
    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Wz: float
        Channel height [m]
    Wy: float 
        Channel width [m]
    Q:  float
        The flux in the channel in [ul/h]
    outV: 2d float array
        array to use for the return
    Returns
    -------
    F:  2d array
        The step matrix (independent on Q)
    dxtD: float 
        The position step multiplied by the diffusion coefficient
    """
    V=poiseuille(Zgrid,Ygrid,Wz,Wy,Q)
    
    if outV is not None:
        outV[:]=V
        
    
    #% Get The step matrix
    dy=Wy/Ygrid
    dz=Wz/Zgrid
    #flatten V
    V=np.ravel(V)
   
    #get Cyy
    udiag=np.ones(Ygrid*Zgrid-1)
    udiag[Ygrid-1::Ygrid]=0
    Cyy=np.diag(udiag,1)+np.diag(udiag,-1)
    Cyy-=np.diag(np.sum(Cyy,0))
    Cyy/=dy**2
    
    #get Czz
    Czz=0
    if Zgrid>1:
        udiag=np.ones(Ygrid*(Zgrid-1))
        Czz=np.diag(udiag,-Ygrid)+np.diag(udiag,Ygrid)
        Czz-=np.diag(np.sum(Czz,0))
        Czz/=dz**2
        
    Lapl=np.dot(np.diag(1/V),Cyy+Czz)
    #get F
    #The formula gives F=1+dx*D/V*(1/dy^2*dyyC+1/dz^2*dzzC)
    #Choosing dx as dx=dy^2*Vmin/D, The step matrix is:
    dxtD=np.min((dy,dz))**2*V.min()/2
    I=np.eye(Ygrid*Zgrid, dtype=float)
    F=I+dxtD*Lapl
    #The maximal eigenvalue should be <=1! otherwhise no stability
    #The above dx should put it to 1
#    from numpy.linalg import eigvals
#    assert(np.max(np.abs(eigvals(F)))<=1.)
    return F, dxtD

def dxtDd(Zgrid,Ygrid,Wz,Wy,Q,outV=None):
    """
    Compute the position step
    
    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Wz: float
        Channel height [m]
    Wy: float 
        Channel width [m]
    Q:  float
        The flux in the channel in [ul/h]
    outV: 2d float array
        array to use for the return
    Returns
    -------
    dxtD: float 
        The position step multiplied by the diffusion coefficient
    """
    V=poiseuille(Zgrid,Ygrid,Wz,Wy,Q,outV)
    #% Get The step matrix
    dy=Wy/Ygrid
    dz=Wz/Zgrid    

    dxtD=np.min((dy,dz))**2*V.min()/2
    return dxtD


#@profile
def getprofiles(Cinit,Q, Radii, readingpos,  Wy = 300e-6, Wz= 50e-6, Zgrid=1,
                *,fullGrid=False, outV=None,central_profile=False,
                eta=1e-3, kT=1.38e-23*295):
    """Returns the theorical profiles for the input variables
    
    Parameters
    ----------
    Cinit:  1d array or 2d array
            The initial profile. If 1D (shape (x,) not (x,1)) Zgrid is 
            used to pad the array
    Q:  float
        The flux in the channel in [ul/h]
    Radii: 1d array
        The simulated radius. Must be in increasing order [m]
    readingpos: 1d array float
        Position to read at
    Wy: float, defaults 300e-6 
        Channel width [m]
    Wz: float, defaults 50e-6
        Channel height [m]  
    Zgrid:  integer, defaults 1
        Number of Z pixel if Cinit is unidimentional
    fullGrid: bool , false
        Should return full grid?
    outV: 2d float array
        array to use for the poiseuiile flow
    central_profile: Bool, default False
        If true, returns only the central profile
    eta: float
        eta
    kT: float
        kT

    Returns
    -------
    profilespos: 3d array
        The list of profiles for the 12 positions at the required radii
    
    """    
    Radii=np.array(Radii)
    assert not np.any(Radii<0), "Can't work with negative radii!"
    #Functions to access F
    def getF(Fdir,NSteps):
        if NSteps not in Fdir:
            Fdir[NSteps]=np.dot(Fdir[NSteps//2],Fdir[NSteps//2])
        return Fdir[NSteps]  
    def initF(Zgrid,Ygrid,Wz,Wy,Q,outV):
        key=(Zgrid,Ygrid,Wz,Wy)
        if not hasattr(getprofiles,'dirFList') :
            getprofiles.dirFList = {}
        #Create dictionnary if doesn't exist
        if key in getprofiles.dirFList:
            return getprofiles.dirFList[key], dxtDd(*key,Q,outV)
        else:
            Fdir={}
            Fdir[1],dxtd=stepMatrix(Zgrid,Ygrid,Wz,Wy,Q,outV)
            getprofiles.dirFList[key]=Fdir
            return Fdir,dxtd
        
    #Prepare input and Initialize arrays
    readingpos=np.asarray(readingpos)
    
    Cinit=np.asarray(Cinit,dtype=float)
    if len(Cinit.shape)<2:
        Cinit=np.tile(Cinit[:,np.newaxis],(1,Zgrid)).T
        
    Ygrid = Cinit.shape[1];
    NRs=len(Radii)
    Nrp=len(readingpos)
    profilespos=np.tile(np.ravel(Cinit),(NRs*Nrp,1))
    
    #get step matrix
    Fdir,dxtD=initF(Zgrid,Ygrid,Wz,Wy,Q,outV)       

    #Get Nsteps for each radius and position
    Nsteps=np.empty((NRs*Nrp,),dtype=int)         
    for i,r in enumerate(Radii):
        D = kT/(6*np.pi*eta*r)
        dx=dxtD/D
        Nsteps[Nrp*i:Nrp*(i+1)]=np.asarray(readingpos//dx,dtype=int)
     
    print('{} steps'.format(Nsteps.max()))
    #transform Nsteps to binary array
    pow2=1<<np.arange(int(np.floor(np.log2(Nsteps.max())+1)))
    pow2=pow2[:,None]
    binSteps=np.bitwise_and(Nsteps[None,:],pow2)>0
    
    #Sort for less calculations
    sortedbs=np.argsort([str(num) for num in np.asarray(binSteps,dtype=int).T])
    
    #for each unit
    for i,bsUnit in enumerate(binSteps):
        F=getF(Fdir,2**i)
            
#        print("NSteps=%d" % 2**i)
        #save previous number
        prev=np.zeros(i+1,dtype=bool)
        for j,bs in enumerate(bsUnit[sortedbs]):#[sortedbs]
            prof=profilespos[sortedbs[j],:]
            act=binSteps[:i+1,sortedbs[j]]
            #If we have a one, multiply by the current step function
            if bs:
                #If this is the same as before, no need to recompute
                if (act==prev).all():
                    prof[:]=profilespos[sortedbs[j-1]]
                else:
                    prof[:]=np.dot(F,prof)
            prev=act
         
    #reshape correctly
    profilespos.shape=(NRs,Nrp,Zgrid,Ygrid)
    
    #If full grid, stop here
    if fullGrid:
        return profilespos
    
    if central_profile:
        #Take central profile
        central_idx=int((Zgrid-1)/2)
        profilespos=profilespos[:,:,central_idx,:]
    else:
        #Take mean
        profilespos=np.mean(profilespos,-2)
    
    #Normalize to avoid mass destruction / creation
    profilespos/=np.sum(profilespos,-1)[:,:,np.newaxis]/np.sum(Cinit/Zgrid)
    
    return profilespos
#%%        
def stepMatrixElectro(Zgrid,Ygrid,Wz,Wy,Q,D,muE,outV=None):
    """
    Compute the step matrix and corresponding position step
    
    Parameters
    ----------
    Zgrid:  integer
        Number of Z pixel
    Ygrid:  integer
        Number of Y pixel
    Wz: float
        Channel height [m]
    Wy: float 
        Channel width [m]
    Q:  float
        The flux in the channel in [ul/h]
    D:  float
        The diffusion coefficient
    muE: float
        the convective speed
    outV: 2d float array
        array to use for the return
    Returns
    -------
    F:  2d array
        The step matrix (independent on Q)
    dx: float 
        The position step 
    """
    
    #% Get The step matrix
    dy=Wy/Ygrid
#    dz=Wz/Zgrid
    #flatten V
    Vx=Q/(3600*1e9)/Wy/Wz
#    V=poiseuille(Zgrid,Ygrid,Wz,Wy,Q,outV)
#    Vx=np.ravel(V)
    
    #get Dyy
    line=np.zeros(Ygrid*Zgrid)
    line[:2]=[-2,1]
    Dyy=toeplitz(line,line) #toeplitz creation of matrice which repeat in diagonal 
    for i in range(0,Ygrid*Zgrid,Ygrid):
        Dyy[i,i]=-1
        Dyy[i-1+Ygrid,i-1+Ygrid]=-1
        if i>0 :
            Dyy[i-1,i]=0
            Dyy[i,i-1]=0

            
    #get Dy
    if muE>0:
        Dy=np.diag(np.ones(Ygrid*Zgrid),0)+np.diag(-np.ones(Ygrid*Zgrid-1),-1)
    else:
        Dy=np.diag(np.ones(Ygrid*Zgrid-1),1)+np.diag(-np.ones(Ygrid*Zgrid),0)
#    Dy=np.diag(np.ones(Ygrid*Zgrid-1),1)+np.diag(-np.ones(Ygrid*Zgrid-1),-1)
#    Dy=Dy/2    
    for i in range(0,Ygrid*Zgrid,Ygrid):
        Dy[i,i]=0
        Dy[i-1+Ygrid,i-1+Ygrid]=0
        if i>0 :
            Dy[i-1,i]=0
            Dy[i,i-1]=0
    Dy/=(dy)
    #get F
    #The formula gives F=1+dx*D/V*(1/dy^2*dyyC+1/dz^2*dzzC)
    #Choosing dx as dx=dy^2*Vmin/D, The step matrix is:
    F=1/dy**2*Dyy#+1/dz**2*Dzz
#    dx=Vx.min()/(2*D/(np.min((dy,dz))**2)+muE/(dy))/2
    dx=np.nanmin([dy*np.min(Vx)/muE,np.min(Vx)*dy**2/D/2])/2

    I=np.eye(Ygrid*Zgrid, dtype=float)
    F=I+dx*1/Vx*(D*F-muE*Dy)#
    

    #The maximal eigenvalue should be <=1! otherwhise no stability
    #The above dx should put it to 1
#    from numpy.linalg import eigvals
#    assert(np.max(np.abs(eigvals(F)))<=1.)
    return F, dx

