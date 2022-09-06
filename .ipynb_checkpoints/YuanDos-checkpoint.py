import cmath
import numpy as np
from mpmath import ellipk
from scipy.special import gamma as GAMMA


def YuanDoS(E,Ev,alpha,beta,gamma,kappa, eta = 1e-7): 
    '''Returns Yuan DoS Eq.5 given the dispersion parameters'''
    
    if (gamma**2 + 4*alpha*kappa) < 0: 
        raise(Exception("Invalid parameter regime"))

    gammatwid = np.sqrt(gamma**2 + 4*alpha*kappa)
    epsilon = (gammatwid**2) * (E - Ev + 1j*eta) / (alpha**3)
    zplus = (beta/alpha) + cmath.sqrt((beta/alpha)**2 + epsilon)
    if not zplus: 
        raise(Exception("ERROR!! zplus cannot be zero, E = " + str(E)))

    zminus = (beta/alpha) - cmath.sqrt((beta/alpha)**2 + epsilon)
    if not zminus: 
        raise(Exception("ERROR!! zminus cannot be zero, E = " + str(E)))


    sign = 1 - 2*np.heaviside(-1.0*alpha*beta,0)
    pref = sign/(np.sqrt(2)*alpha*(np.pi**2))

    term1 = (1/cmath.sqrt(zminus)) * ellipk(np.real(1 - (zplus/zminus)))
    term2 = (2j/cmath.sqrt(zplus)) * ellipk(np.real(zminus/zplus)) * np.heaviside(-1.0*alpha*beta,0)
    DoS = pref * np.real(term1 - term2)
    return DoS


def DiracConeDoS_TBLG(e, angle, w=0.11, hbar_vF_divided_a=4.31075):
    '''This function uses exact expression for Dirac cone dispersion in TBLG to compute DOS
    
    Parameters: 
    e - energy in eV
    angle - in radians
    w - tunneling parameters
    hbar_vF_divided_a - parameter of graphene Fermi velocity in eV
    '''
    alpha_par = w*3*np.sqrt(3)/(8*np.pi*hbar_vF_divided_a*np.sin(angle/2))
    vFeff = ((1-3*alpha_par**2)/(1+6*alpha_par**2)) #*0.0031
    return 2*abs(e)/(np.pi*vFeff**2)


def LogvHSDoS(e,alpha,beta,Ev,offset=0, cutoffscale=1):
    '''This function computes the DOS coming out of a logarithmic vHS governed by a hyperbolic dispersion
    
    Parameters:
    e - energy in eV
    alpha - coefficient of the minimumum direction (px**2) 
    beta - coefficient of the maximum direction (py**2)
    Ev - the saddle point energy in eV
    offset - shifts the entire function by a constant
    cutoffscale - the cutoff in the log: plays a similar role as the offset
    '''
    pref = 1.0/(4*np.pi**2)
    pref *= 1.0/np.sqrt(np.abs(alpha*beta))
    return pref * 1.0*np.log(np.abs(cutoffscale/(e-Ev))) + offset


def MagicvHSDoS(e,alpha,gamma,kappa,offset=0):
    '''This function computes the DOS exactly at the magic angle when beta=0
    
    Parameters:
    e - energy in eV
    alpha - coefficient of the minimumum direction (px**2) 
    gamma - coefficient of the px py**2 term (order 3)
    kappa - coefficient of the py**4 term (order 4)
    '''
    Gammatwidsquare = gamma**2 - 4*alpha*kappa 
    pref = ((2*np.pi)**(-2.5)) * (GAMMA(0.25)**2)
    denom = (4.0*alpha*Gammatwidsquare)**0.25
    return (pref/denom) * e**(-0.25) + offset


def fermidirac(e,s=1):
    '''This function is used as a proxy to perform the smoothing needed.
    
    Parameters: 
    e - energy, 
    s - smoothing parameter, plays the role of a fictitious inverse temperature
    '''
    return (1.0/(1+np.exp(s*e)))