import autograd.numpy as np

def JUGAAD_generate_spectrum_Twisted_graphene_single_point(vF, w, Theta, kx_fixed=0.0, ky_fixed=0.0, symmetric=True):
    '''spectrum data for 2D plots'''
    kx = kx_fixed
    ky = ky_fixed
    
    kTheta = 8.0 * np.pi * np.sin(Theta/2.0) / (3*np.sqrt(3.0))
    onex = 1.0
    
    hamiltonians_TBLG = np.zeros((8, 8), dtype=complex)
    
    if symmetric:
        #no phases with theta
        hamiltonians_TBLG[0, 1] = vF*(kx + 1j*ky)
        hamiltonians_TBLG[1, 0] = vF*(kx - 1j*ky)
        hamiltonians_TBLG[2, 3] = vF*(kx + 1j*(ky+onex*kTheta))
        hamiltonians_TBLG[3, 2] = vF*(kx - 1j*(ky+onex*kTheta))
        hamiltonians_TBLG[4, 5] = vF*(kx-0.5*np.sqrt(3.0)*onex*kTheta+1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[5, 4] = vF*(kx-0.5*np.sqrt(3.0)*onex*kTheta-1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[6, 7] = vF*(kx+0.5*np.sqrt(3.0)*onex*kTheta+1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[7, 6] = vF*(kx+0.5*np.sqrt(3.0)*onex*kTheta-1j*(ky-0.5*onex*kTheta))
    else:    
        hamiltonians_TBLG[0, 1] = vF*(kx + 1j*ky)*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[1, 0] = vF*(kx - 1j*ky)*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[2, 3] = vF*(kx + 1j*(ky+onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[3, 2] = vF*(kx - 1j*(ky+onex*kTheta))*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[4, 5] = vF*(kx-0.5*np.sqrt(3.0)*onex*kTheta+1j*(ky-0.5*onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[5, 4] = vF*(kx-0.5*np.sqrt(3.0)*onex*kTheta-1j*(ky-0.5*onex*kTheta))*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[6, 7] = vF*(kx+0.5*np.sqrt(3.0)*onex*kTheta+1j*(ky-0.5*onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[7, 6] = vF*(kx+0.5*np.sqrt(3.0)*onex*kTheta-1j*(ky-0.5*onex*kTheta))*np.exp(1j*Theta/2.0)
    
    hamiltonians_TBLG[0, 2] = w*onex
    hamiltonians_TBLG[0, 3] = w*onex
    hamiltonians_TBLG[1, 2] = w*onex
    hamiltonians_TBLG[1, 3] = w*onex
    
    hamiltonians_TBLG[0, 4] = w*onex
    hamiltonians_TBLG[0, 5] = w*(-0.5+1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[1, 4] = w*(-0.5-1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[1, 5] = w*onex
    
    hamiltonians_TBLG[0, 6] = w*onex
    hamiltonians_TBLG[0, 7] = w*(-0.5-1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[1, 6] = w*(-0.5+1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[1, 7] = w*onex
    
    #vertical blocks are the same
    hamiltonians_TBLG[2, 0] = w*onex
    hamiltonians_TBLG[3, 0] = w*onex
    hamiltonians_TBLG[2, 1] = w*onex
    hamiltonians_TBLG[3, 1] = w*onex
    
    hamiltonians_TBLG[4, 0] = w*onex
    hamiltonians_TBLG[4, 1] = w*(-0.5+1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[5, 0] = w*(-0.5-1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[5, 1] = w*onex
    
    hamiltonians_TBLG[6, 0] = w*onex
    hamiltonians_TBLG[6, 1] = w*(-0.5-1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[7, 0] = w*(-0.5+1j*np.sqrt(3.0)/2.0)*onex
    hamiltonians_TBLG[7, 1] = w*onex
    
    
    spectrum_data = np.linalg.eigvalsh(hamiltonians_TBLG)
    
    return spectrum_data