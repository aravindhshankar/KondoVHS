import numpy as np
from scipy.linalg import eigh
from SourcesTBG import *
from MoreSources import *


def generate_pdvx_Twisted_graphene_single_point(vF, w, Theta, kx_fixed=0.0, ky_fixed=0.0, symmetric=True):
    '''spectrum data for 2D plots'''
    kx = kx_fixed
    ky = ky_fixed
    
    kTheta = 8 * np.pi * np.sin(Theta/2.0) / (3*np.sqrt(3))
    onex = 1.0
    
    hamiltonians_TBLG = np.zeros((8, 8), dtype=complex)
    
    #no phases with theta
    hamiltonians_TBLG[0, 1] = vF*(onex)
    hamiltonians_TBLG[1, 0] = vF*(onex)
    hamiltonians_TBLG[2, 3] = vF*(onex)
    hamiltonians_TBLG[3, 2] = vF*(onex)
    hamiltonians_TBLG[4, 5] = vF*(onex)
    hamiltonians_TBLG[5, 4] = vF*(onex)
    hamiltonians_TBLG[6, 7] = vF*(onex)
    hamiltonians_TBLG[7, 6] = vF*(onex)
    
    return hamiltonians_TBLG

def generate_pdvy_Twisted_graphene_single_point(vF, w, Theta, kx_fixed=0.0, ky_fixed=0.0, symmetric=True):
    '''spectrum data for 2D plots'''
    kx = kx_fixed
    ky = ky_fixed
    
    kTheta = 8 * np.pi * np.sin(Theta/2.0) / (3*np.sqrt(3))
    onex = 1.0
    
    hamiltonians_TBLG = np.zeros((8, 8), dtype=complex)
    
    if symmetric:
        #no phases with theta
        hamiltonians_TBLG[0, 1] = vF*(1j)
        hamiltonians_TBLG[1, 0] = vF*(-1j)
        hamiltonians_TBLG[2, 3] = vF*(1j)
        hamiltonians_TBLG[3, 2] = vF*(-1j)
        hamiltonians_TBLG[4, 5] = vF*(1j)
        hamiltonians_TBLG[5, 4] = vF*(-1j)
        hamiltonians_TBLG[6, 7] = vF*(1j)
        hamiltonians_TBLG[7, 6] = vF*(-1j)  
    
    return hamiltonians_TBLG


def FHpdvs(vF=vFpar, w=wpar, Theta=1.05*IN_RADIAN, kx_fixed=0.0, ky_fixed=0.0, symmetric=True):
    '''pass Theta argument as thetai * IN_RADIAN'''
    ham = return_hamiltonian_Twisted_graphene_single_point(vFpar, wpar, Theta=Theta, 
                                                           kx_fixed=kx_fixed, ky_fixed=ky_fixed, symmetric=True)
    energ, reqvec = eigh(ham, subset_by_index=[4,4])
    energ = np.real(energ)
    reqvec = reqvec.flatten()
    pdvxmat = generate_pdvx_Twisted_graphene_single_point(vFpar, wpar, Theta=Theta, 
                                                          kx_fixed=kx_fixed, ky_fixed=ky_fixed, symmetric=True)
    reqpdvx = np.conjugate(reqvec)@pdvxmat@reqvec
    reqpdvx = np.real(reqpdvx)
    pdvymat = generate_pdvy_Twisted_graphene_single_point(vFpar, wpar, Theta=Theta, 
                                                          kx_fixed=kx_fixed, ky_fixed=ky_fixed, symmetric=True)
    reqpdvy = np.conjugate(reqvec)@pdvymat@reqvec
    reqpdvy = np.real(reqpdvy)
    return np.array([reqpdvx,reqpdvy])


def FHfuncfsquare(xG, vF=vFpar, w=wpar, Theta=1.05*IN_RADIAN, symmetric=True):
    kxval,kyval = xG
    FHpdvx, FHpdvy = FHpdvs(vF = vFpar, w = wpar, Theta = Theta, kx_fixed = kxval, ky_fixed = kyval)
    return 1e6 * (FHpdvx**2 + FHpdvy**2)