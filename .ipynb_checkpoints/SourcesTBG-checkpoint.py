import numpy as np

def generate_spectrum_Twisted_graphene_single_point(vF, w, Theta, kx_fixed=0.0, ky_fixed=0.0, symmetric=True):
    '''spectrum data for 2D plots'''
    kx = kx_fixed
    ky = ky_fixed
    
    kTheta = 8 * np.pi * np.sin(Theta/2.0) / (3*np.sqrt(3))
    onex = 1.0
    
    hamiltonians_TBLG = np.zeros((8, 8), dtype=complex)
    
    if symmetric:
        #no phases with theta
        hamiltonians_TBLG[0, 1] = vF*(kx + 1j*ky)
        hamiltonians_TBLG[1, 0] = vF*(kx - 1j*ky)
        hamiltonians_TBLG[2, 3] = vF*(kx + 1j*(ky+onex*kTheta))
        hamiltonians_TBLG[3, 2] = vF*(kx - 1j*(ky+onex*kTheta))
        hamiltonians_TBLG[4, 5] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[5, 4] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[6, 7] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[7, 6] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))
    else:    
        hamiltonians_TBLG[0, 1] = vF*(kx + 1j*ky)*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[1, 0] = vF*(kx - 1j*ky)*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[2, 3] = vF*(kx + 1j*(ky+onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[3, 2] = vF*(kx - 1j*(ky+onex*kTheta))*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[4, 5] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[5, 4] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[6, 7] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[7, 6] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))*np.exp(1j*Theta/2.0)
    
    hamiltonians_TBLG[0, 2] = w*onex
    hamiltonians_TBLG[0, 3] = w*onex
    hamiltonians_TBLG[1, 2] = w*onex
    hamiltonians_TBLG[1, 3] = w*onex
    
    hamiltonians_TBLG[0, 4] = w*onex
    hamiltonians_TBLG[0, 5] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 4] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 5] = w*onex
    
    hamiltonians_TBLG[0, 6] = w*onex
    hamiltonians_TBLG[0, 7] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 6] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 7] = w*onex
    
    #vertical blocks are the same
    hamiltonians_TBLG[2, 0] = w*onex
    hamiltonians_TBLG[3, 0] = w*onex
    hamiltonians_TBLG[2, 1] = w*onex
    hamiltonians_TBLG[3, 1] = w*onex
    
    hamiltonians_TBLG[4, 0] = w*onex
    hamiltonians_TBLG[4, 1] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[5, 0] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[5, 1] = w*onex
    
    hamiltonians_TBLG[6, 0] = w*onex
    hamiltonians_TBLG[6, 1] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[7, 0] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[7, 1] = w*onex
    
    
    spectrum_data = np.linalg.eigvalsh(hamiltonians_TBLG)
    
    return spectrum_data





def derivative(f,x,epsilon = 1e-8, maxiter = 1000,hstart = 2**-3,conv = True):
    '''f is a callable that returns f(x)'''
    if conv:
        h = hstart
        der = (f(x+h/2.0) - f(x-h/2.0))/h
        oldder = der
        diffs = 1
        itern = 0
        wrench = 1.0
        #h /= 2
        while(diffs > epsilon and itern <= maxiter):
            h /= 2
            olddiffs = diffs
            der = wrench*((f(x+h/2.0) - f(x-h/2.0))/h) + (1-wrench)*oldder
            diffs = np.abs(oldder - der)
            oldder = der
            itern += 1
            if diffs > olddiffs:
                wrench /= 2.0
                #h /= 2
                
            if itern > maxiter:
                Error_Message = "Derivative did not converge in " + str(itern) + " iterations\n"
                Error_Message += " x = " + str(x) + " h = " + str(h) 
                Error_Message += " wrench = " + str(wrench)
                Error_Message += " diffs = " + str(diffs)
                print(Error_Message)
                #der = 1e5
        #print(" wrench = " + str(wrench))
        #print(" itern = "  + str(itern))

        return der
    else:
        h = hstart
        der = (f(x+h/2.0) - f(x-h/2.0))/h
        oldder = der
        diffs = 1
        itern = 0
        while(diffs > epsilon and itern <= maxiter):
            h /= 2
            der = (f(x+h/2.0) - f(x-h/2.0))/h
            diffs = np.abs(oldder - der)
            oldder = der
            itern += 1
            if itern > maxiter:
                Error_Message = "Derivative did not converge in " + str(itern) + " iterations\n"
                Error_Message += "x = " + str(x) + " h = " + str(h)
                print(Error_Message)
                der = 1e5

        return der

    
    
def return_hamiltonian_Twisted_graphene_single_point(vF, w, Theta, kx_fixed=0.0, ky_fixed=0.0, symmetric=True):
    '''spectrum data for 2D plots'''
    kx = kx_fixed
    ky = ky_fixed
    
    kTheta = 8 * np.pi * np.sin(Theta/2.0) / (3*np.sqrt(3))
    onex = 1.0
    
    hamiltonians_TBLG = np.zeros((8, 8), dtype=complex)
    
    if symmetric:
        #no phases with theta
        hamiltonians_TBLG[0, 1] = vF*(kx + 1j*ky)
        hamiltonians_TBLG[1, 0] = vF*(kx - 1j*ky)
        hamiltonians_TBLG[2, 3] = vF*(kx + 1j*(ky+onex*kTheta))
        hamiltonians_TBLG[3, 2] = vF*(kx - 1j*(ky+onex*kTheta))
        hamiltonians_TBLG[4, 5] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[5, 4] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[6, 7] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))
        hamiltonians_TBLG[7, 6] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))
    else:    
        hamiltonians_TBLG[0, 1] = vF*(kx + 1j*ky)*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[1, 0] = vF*(kx - 1j*ky)*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[2, 3] = vF*(kx + 1j*(ky+onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[3, 2] = vF*(kx - 1j*(ky+onex*kTheta))*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[4, 5] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[5, 4] = vF*(kx-0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))*np.exp(1j*Theta/2.0)
        hamiltonians_TBLG[6, 7] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta+1j*(ky-0.5*onex*kTheta))*np.exp(-1j*Theta/2.0)
        hamiltonians_TBLG[7, 6] = vF*(kx+0.5*np.sqrt(3)*onex*kTheta-1j*(ky-0.5*onex*kTheta))*np.exp(1j*Theta/2.0)
    
    hamiltonians_TBLG[0, 2] = w*onex
    hamiltonians_TBLG[0, 3] = w*onex
    hamiltonians_TBLG[1, 2] = w*onex
    hamiltonians_TBLG[1, 3] = w*onex
    
    hamiltonians_TBLG[0, 4] = w*onex
    hamiltonians_TBLG[0, 5] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 4] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 5] = w*onex
    
    hamiltonians_TBLG[0, 6] = w*onex
    hamiltonians_TBLG[0, 7] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 6] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[1, 7] = w*onex
    
    #vertical blocks are the same
    hamiltonians_TBLG[2, 0] = w*onex
    hamiltonians_TBLG[3, 0] = w*onex
    hamiltonians_TBLG[2, 1] = w*onex
    hamiltonians_TBLG[3, 1] = w*onex
    
    hamiltonians_TBLG[4, 0] = w*onex
    hamiltonians_TBLG[4, 1] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[5, 0] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[5, 1] = w*onex
    
    hamiltonians_TBLG[6, 0] = w*onex
    hamiltonians_TBLG[6, 1] = w*(-0.5-1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[7, 0] = w*(-0.5+1j*np.sqrt(3)/2.0)*onex
    hamiltonians_TBLG[7, 1] = w*onex
    
    return hamiltonians_TBLG
    
# def derivative(f,x,epsilon = 1e-8, maxiter = 100,hstart = 2**-3):
#     '''f is a callable that returns f(x)'''
#     h = hstart
#     der = (f(x+h/2.0) - f(x-h/2.0))/h
#     oldder = der
#     diffs = 1
#     itern = 0
#     while(diffs > epsilon and itern <= maxiter):
#         h /= 2
#         der = (f(x+h/2.0) - f(x-h/2.0))/h
#         diffs = np.abs(oldder - der)
#         oldder = der
#         itern += 1
#         if itern > maxiter:
#             Error_Message = "Derivative did not converge in " + str(itern) + " iterations\n"
#             Error_Message += "x = " + str(x) + " h = " + str(h)
#             print(Error_Message)
#             der = 1e5
            
#     return der