import sys, platform, os
sys.path.insert(1, '/home1/jacklone/Mat_project/hmvec/hmvec')

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower
from scipy.interpolate import interp1d, interp2d
from params import default_params
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

defaults = {'min_mass':1e6, 'max_mass':1e16, 'num_mass':1000}
constants = {
    'thompson_SI': 6.6524e-29,
    'meter_to_megaparsec': 3.241e-23,
    'G_SI': 6.674e-11,
    'mProton_SI': 1.673e-27,
    'H100_SI': 3.241e-18
}

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

#Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
#Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)

#Linear spectra
results = camb.get_results(pars)
k, z, p = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
growth = results.get_redshift_evolution(k, z, ['growth'])[:,:,0]

#############################################################################################################
# Constructing the covariance matrix C
# Starting with the signal covariance matrix S
# P_vv component:
f = []
a = []
H = []
d2vs = []
for i in range(len(z)):
	f.append(growth[:,i])
	a.append(1./1. + z[i])
	H.append(results.h_of_z(z[i]))
	d2vs.append((f[i]*a[i]*H[i])/k[i])
	       
bv1 = 1
bv2 = 1
P_vv = []
for i in range(len(z)):
	P_vv.append(d2vs[i]**2*p[i]*bv1*bv2)

# Defining the sigma(z) function
sigz = 0.03
sigma_z_func = lambda z : 0.03*(1.0 + z)

# P_vg component:
bv = 1
bg = 1
P_vg = []
Wphoto = []
for i in range(len(z)):
	Wphoto.append(np.exp(-sigma_z_func(z[i])**2.*k[i]**2./2./H[i]**2))
	P_vg.append(d2vs[i]*p[i]*bv*bg*Wphoto[i])
P_gv = P_vg

# P_gg component:
bg1 = 1
bg2 = 1
P_gg = []
for i in range(len(z)):
	Wphoto.append(np.exp(-sigma_z_func(z[i])**2.*k[i]**2./2./H[i]**2))
	P_gg.append(p[i]*bg1*bg2*Wphoto[i])

# Putting elements into matrix:
S = [[P_vv, P_vg], [P_gv, P_gg]]

# Constructing the noise covariance matrix N
# Nvv component:
def _sanitize(inp):
    inp[~np.isfinite(inp)] = 0
    return inp

def get_interpolated_cls(Cls,chistar,kss):
    ls = np.arange(Cls.size)
    Cls[ls<2] = 0
    def _Cls(ell):
        if ell<=ls[-1]:
            return Cls[int(ell)]
        else:
            return np.inf
    # TODO: vectorize
    return np.array([_Cls(chistar*k) for k in kss])

def get_kmin(volume_gpc3):
    vol_mpc3 = volume_gpc3 * 1e9
    return np.pi/vol_mpc3**(1./3.)

def chi(Yp,NHe):
    val = (1-Yp*(1-NHe/4.))/(1-Yp/2.)
    return val

def ne0_shaw(ombh2,Yp,NHe=0,me = 1.14,gasfrac = 0.9):
    '''
    Average electron density today
    Eq 3 of 1109.0553
    Units: 1/meter**3
    '''
    omgh2 = gasfrac* ombh2
    mu_e = 1.14 # mu_e*mass_proton = mean mass per electron
    ne0_SI = chi(Yp,NHe)*omgh2 * 3.*(constants['H100_SI']**2.)/constants['mProton_SI']/8./np.pi/constants['G_SI']/mu_e
    return ne0_SI

def ksz_radial_function(z,ombh2, Yp, gasfrac = 0.9,xe=1, tau=0, params=None):
    """
    K(z) = - T_CMB sigma_T n_e0 x_e(z) exp(-tau(z)) (1+z)^2
    Eq 4 of 1810.13423
    """
    if params is None: params = default_params
    T_CMB_muk = params['T_CMB'] # muK
    thompson_SI = constants['thompson_SI']
    meterToMegaparsec = constants['meter_to_megaparsec']
    ne0 = ne0_shaw(ombh2,Yp)
    return T_CMB_muk*thompson_SI*ne0*(1.+z)**2./meterToMegaparsec  * xe  *np.exp(-tau)

num_mu_bins = 102
mu = np.linspace(-1,1,num_mu_bins)
chistars = []
Fstar = []
for i in range(len(z)):
    chistars.append(results.comoving_radial_distance(z[i]))
    Fstar.append(ksz_radial_function(z = z[i], ombh2 = pars.ombh2, Yp = pars.YHe)) 

kLs = np.geomspace(get_kmin(np.max(volumes_gpc3)),kL_max,num_kL_bins)

def Nvv_core_integral(chi_star,Fstar,mu,kL,kSs,Cls,Pge,Pgg_tot,Pgg_photo_tot=None,errs=False,
                      robust_term=False,photo=True):
    """
    Returns velocity recon noise Nvv as a function of mu,kL
    Uses Pgg, Pge function of mu,kL,kS and integrates out kS
    if errs is True: sets Pge=1, so can be reused for Pge error calc
    Cls is an array for C_tot starting at l=0.
    e.g. C_tot = C_CMB + C_fg + (C_noise/beam**2 )
    """

    if robust_term:
        if photo: print("WARNING: photo_zs were True for an Nvv(robust_term=True) call. Overriding to False.")
        photo = False

    if errs:
        ret_Pge = Pge.copy()
        Pge = 1.

    amu = np.resize(mu,(kL.size,mu.size)).T
    prefact = amu**(-2.) * 2. * np.pi * chi_star**2. / Fstar**2.


   # Clkstot = get_interpolated_cls(Cls,chi_star,kSs)
    Clkstot = Cls
    integrand = _sanitize(kSs * ( Pge**2. / (Pgg_tot * Clkstot)))

    if robust_term:
        assert Pgg_photo_tot is not None
        integrand = _sanitize(integrand * (Pgg_photo_tot/Pgg_tot))

    integral = np.trapz(integrand,kSs)
    Nvv = prefact / integral
    assert np.all(np.isfinite(Nvv))
    if errs:
        return Nvv,ret_Pge
    else:
        return Nvv
