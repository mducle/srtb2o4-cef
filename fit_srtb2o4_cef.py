# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import copy
import scipy

from CrystalField import CrystalField, PointCharge, ResolutionModel, CrystalFieldFit, Background, Function
from CrystalField.energies import energies
from pychop.Instruments import Instrument

sys.path.append(os.path.dirname(__file__))
import cef_utils
import importlib
importlib.reload(cef_utils)

np.set_printoptions(linewidth=200)

#%%
##########################
# Setup

# Measured energy levels for each site to refine to:
e0 = [[0, 0.7, 7.5, 30.35, 32.2], [0, 1.3, 12.2, 31.0]]

# Conversion factor from Wybourne to Stevens normalisation
from scipy import sqrt
lambdakq = {'IB22':sqrt(6.)/2., 'IB21':sqrt(6.), 'B20':1./2., 'B21':sqrt(6.), 'B22':sqrt(6.)/2.,
     'IB44':sqrt(70.)/8., 'IB43':sqrt(35.)/2., 'IB42':sqrt(10.)/4., 'IB41':sqrt(5.)/2., 'B40':1./8., 'B41':sqrt(5.)/2., 'B42':sqrt(10.)/4., 'B43':sqrt(35.)/2., 'B44':sqrt(70.)/8., 
     'IB66':sqrt(231.)/16., 'IB65':3*sqrt(77.)/8., 'IB64':3*sqrt(14.)/16., 'IB63':sqrt(105.)/8., 'IB62':sqrt(105.)/16., 'IB61':sqrt(42.)/8.,
     'B60':1./16., 'B61':sqrt(42.)/8., 'B62':sqrt(105.)/16., 'B63':sqrt(105.)/8., 'B64':3*sqrt(14.)/16., 'B65':3*sqrt(77.)/8., 'B66':sqrt(231.)/16.}

# Stevens Operator Equivalent factors
idx = {'IB22':0, 'IB21':0, 'B20':0, 'B21':0, 'B22':0, 'IB44':1, 'IB43':1, 'IB42':1, 'IB41':1, 'B40':1, 'B41':1, 'B42':1, 'B43':1, 'B44':1, 
       'IB66':2, 'IB65':2, 'IB64':2, 'IB63':2, 'IB62':2, 'IB61':2, 'B60':2, 'B61':2, 'B62':2, 'B63':2, 'B64':2, 'B65':2, 'B66':2}
thetakq = {'Tb': [-1.0 * 1/3/3/11, 1.0 * 2/3/3/3/5/11/11, -1.0 * 1/3/3/3/3/7/11/11/13],
           'Dy': [-1.0 * 2/3/3/5/7, -1.0 * 2*2*2/3/3/3/5/7/11/13, 1.0 * 2*2/3/3/3/7/11/11/13/13],
           'Ho': [-1.0 * 1/2/3/3/5/5, -1.0 * 1/2/3/5/7/11/13, -1.0 * 5/3/3/3/7/11/11/13/13],
           'Er': [1.0 * 2*2/3/3/5/5/7, 1.0 * 2/3/3/5/7/11/13, 1.0 * 2*2*2/3/3/3/7/11/11/13/13],
           'Tm': [1.0 * 1/3/3/11, 1.0 * 2*2*2/3/3/3/3/5/11/11, -1.0 * 5/3/3/3/3/7/11/11/13]}


# Malkin PRB 92 094415 (2015) - parameters for SrY2O4:Er and SrEr2O4 (in "Standard" Stevens normalisation (with Stevens factor) in cm^-1)
malkin_pars_order = ['B20', 'B22', 'IB22', 'B40', 'B42', 'IB42', 'B44', 'IB44', 'B60', 'B62', 'IB62', 'B64', 'IB64', 'B66', 'IB66']
malkin_pars_er_R1 = [188, 137.5, -171.2, -57.3, -1066.2, 1165.2, -86.9, -972.3, -38, -22.3, 22.8, 30.1, -115.2, -162.2, -84]
malkin_pars_er_R2 = [17, -744, -125, -60.2, 1033.2, -977.8, 430.2, -685.6, -35.2, -68.4, -42.8, -80.2, -191.4, -119.6, 80.5]

# Nikitin Opt. i Spek. 131 441 (2023) - parameters for SrY2O4:Ho  (in "Standard" Stevens normalisation (with Stevens factor) in cm^-1)
nikitin_pars_ho_R1 = [200.3, 143.1, -142.6, -59.45, -1068.3, 1186.6, -62.4, -942, -40.95, -22.1, 23.1, 3.8, -151.1, -155.7, -99.3]
nikitin_pars_ho_R2 = [-8, -748, -133, -63, 1100, -981, 408, -715, -36.9, -70, -37.4, -73, -208, -115, 95]

# Malkin et al., arXiv:2310.18947 - parameters for SrY2O4:Dy  (in "Standard" Stevens normalisation (with Stevens factor) in cm^-1)
malkin_pars_dy_R1 = [181.5, 90.2, -113.7, -64.2, -1074, 1180, -78.7,  -972.5,  -41.7, -40, 43.3,  24.7,  -141.9, -170.3, -89.25]
malkin_pars_dy_R2 = [-5, -729, -145, -64.7, 1103.2, -927.8, 380.2, -770.6, -37.2, -71.4, -37.8, -80.2, -211.4, -119.6, 90.5]

# Scaling of Er/Ho/Dy pars to Tb
tb1 = {k:((v1+v2)/2)*thetakq['Tb'][idx[k]]/8.066 for k,v1,v2 in zip (malkin_pars_order, malkin_pars_er_R1, nikitin_pars_ho_R1)}
tb2 = {k:((v1+v2)/2)*thetakq['Tb'][idx[k]]/8.066 for k,v1,v2 in zip (malkin_pars_order, malkin_pars_er_R2, nikitin_pars_ho_R2)}

merlin = Instrument('MERLIN', 'G', 150.)
merlin.setEi(7.)
resmod1 = ResolutionModel(merlin.getResolution, xstart=-10, xend=6.9, accuracy=0.01)
merlin.setEi(18.)
resmod2 = ResolutionModel(merlin.getResolution, xstart=-10, xend=17.9, accuracy=0.01)
merlin = Instrument('MERLIN', 'G', 300.)
merlin.setEi(82.)
resmod3 = ResolutionModel(merlin.getResolution, xstart=-10, xend=82.9, accuracy=0.01)

resmods = [resmod1, resmod2, resmod3]

curdir = os.path.dirname(__file__)
mer46207_ei7_cut = Load(f'{curdir}/mer46207_ei7_cut.nxs')
mer46207_ei18_cut = Load(f'{curdir}/mer46207_ei18_cut.nxs')
mer46210_ei82_cut = Load(f'{curdir}/mer46210_ei82_cut.nxs')

def physprop_chi2(blms):
    chi2, invchi, mag = (0.0, [], [])
    invchi_dat, mag_dat, invchi_calc, mag_calc = ([37, 27, 22], [1.7, 3.5, 5.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ic0_dat, ic0_calc = ([5, 1, 1], [0.0, 0.0, 0.0])
    dirs = [[1,0,0], [0,1,0], [0,0,1]]
    for ii, blm in enumerate(blms):
        cfo = CrystalField('Tb', 'C2', Temperature=7, **blm)
        for jj in range(3):
            susc = cfo.getSusceptibility(np.array([1, 300]), Hdir=dirs[jj], Inverse=True, Unit='cgs')
            invchi_calc[jj] += susc[1][1]
            mag_calc[jj] += cfo.getMagneticMoment(T=1.5, Hdir=dirs[jj], Hmag=[8, 8], Unit='bohr')[1][1]
            ic0_calc[jj] += susc[1][0]
    chi2 += np.sum(np.array([invchi_calc[jj] - invchi_dat[jj] for jj in range(3)])**2)
    chi2 += 20 * np.sum(np.array([mag_calc[jj] - mag_dat[jj] for jj in range(3)])**2)
    chi2 += 5 * np.sum(np.array([ic0_calc[jj] - ic0_dat[jj] for jj in range(3)])**2)
    return 100*chi2

def cffitobj():
    FWHMs = [np.interp(0, *resmods[irm].model)*1.5 for irm in [0,1,2]]
    cf1 = CrystalField('Tb', 'C2', Temperature=[7]*3, FWHM=FWHMs, **tb1)
    cf2 = CrystalField('Tb', 'C2', Temperature=[7]*3, FWHM=FWHMs, **tb2)
    cf = cf1 + cf2
    cf.PeakShape = 'PseudoVoigt'
    cf.ToleranceIntensity = 20
    cf.constraints('ion0.B20 < 0')
    cf.background = Background(background=Function('LinearBackground', A0=45, A1=0))
    for pn in tb1.keys():
        cf[f'ion0.{pn}'] = tb1[pn]
        cf[f'ion1.{pn}'] = tb2[pn]
    fitobj = CrystalFieldFit(Model=cf, InputWorkspace=[mer46207_ei7_cut, mer46207_ei18_cut, mer46210_ei82_cut],
                          MaxIterations=0, Output='fit')
    return fitobj


#%%
##########################
# Global fit
try:
    fit = cffitobj()
    res = cef_utils.fit_en(fit, e0, fit_alg='global', algorithm='differential_evolution',
        widths_kwargs={'maxfwhm':[0.5, 3.0, 10.0], 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':10}},
        extra_chi2_fun=physprop_chi2)
    print(res)
    print(res.x.tolist())
except:
    pass
bp = np.squeeze(mtd['bestpars'].extractY())
print(bp.tolist())


#%%
##########################
# Calculates physical properties
cef_utils.genpp(fit)
cef_utils.printpars(fit)


#%%
##########################
# Local fit
fit = cffitobj()
chi2bp = cef_utils.fit_en(fit, e0, eval_only=bp, widths_kwargs={'maxfwhm':[0.5, 3.0, 10.0], 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}})

res = cef_utils.fit_en(fit, e0, fit_alg='local', method='Nelder-Mead', jac='3-point', options={'maxiter':500},
    widths_kwargs={'maxfwhm':[0.5, 3.0, 10.0], 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}})


#%%
##########################
# Evaluate best fit and save datafiles

fit = cffitobj()

bp = [-0.15726033447010263, -0.121794439940964, 2.1797488592494495e-05, -0.007151518683324305, -0.0017831448817177019, -1.6397847135697645e-05, 1.8688256499126993e-05, -0.00013551111493153868, -7.870630154628545e-05, 0.24108533148802103, 0.015366318467031181, -0.011718813988565824, -2.0465801552405912e-05, 0.00014723812025898284, 4.703814737894199e-05, 0.04802911719367105, 0.7443884797493974, -0.00028510553611484817, 0.019835257748601773, 0.006378812619379386, 5.635703920303101e-06, 1.550558284346647e-05, 8.009048093650414e-06, 3.597630660736397e-06, 0.15857614665569283, -0.01888627394015916, -0.012397030987216043, 2.4939600883079418e-05, 2.952750144216143e-06, 2.3157675428880223e-06]
#chi2bp = 17952.61036287019

chi2bp = cef_utils.fit_en(fit, e0, eval_only=bp, 
    widths_kwargs={'maxfwhm':[0.5, 3.0, 10.0], 'method':'trust-constr', 'jac':'3-point', 'options':{'maxiter':200}})
print(chi2bp)

cfpars, cfobjs, peaks, intscal, origwidths = cef_utils.parse_cef_func(fit.model.function)
cef_utils.genpp(fit)
cef_utils.printpars(fit)

for ii, ei in enumerate([7, 18, 82]):
    np.savetxt(f'ins_ei{ei}_calc.dat', np.array([mtd[f'fit_Workspace_{ii}'].readX(1), mtd[f'fit_Workspace_{ii}'].readY(1)]).T, header=f'Calculated INS data for SrTb2O4 Ei={ei}meV\nEn Intensity')
icdat = [mtd['invchi_0_x'].readX(0)]
magdat = [mtd['mag_0_x'].readX(0)]
for ii in ['0_x', '0_y', '0_z', '1_x', '1_y', '1_z']:
    icdat.append(mtd[f'invchi_{ii}'].readY(0))
    magdat.append(mtd[f'mag_{ii}'].readY(0))
np.savetxt('invchi.dat', np.array(icdat).T, header='Calculated inverse susceptibility for SrTb2O4 in (mole/emu)\nT(K) site_1_x site_1_y site_1_z site_2_x site_2_y site_2_z')
np.savetxt('mag.dat', np.array(magdat).T, header='Calculated magnetistion for SrTb2O4 in (uB/Tb-ion)\nH(Tesla) site_1_x site_1_y site_1_z site_2_x site_2_y site_2_z')
print('Site 1')
print(cfobjs[0][0].printWavefunction(range(13)))
print('Site 2')
print(cfobjs[1][0].printWavefunction(range(13)))
print(chi2bp)
print(cfobjs[0][0].getEigenvalues())
print(cfobjs[1][0].getEigenvalues())
