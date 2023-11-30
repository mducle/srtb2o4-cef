import mantid
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import copy
import warnings
import scipy.optimize
import re
from CrystalField.energies import _unpack_complex_matrix
from CrystalField import CrystalField
from CrystalField.normalisation import split2range
from CrystalField.fitting import getSymmAllowedParam

import sys, os
sys.path.append(os.path.dirname(__file__))

try:
    import gofit
except ModuleNotFoundError:
    gofit = None

def CFEnergy(nre, **kwargs):
    cfe = AlgorithmManager.create('CrystalFieldEnergies')
    cfe.initialize()
    cfe.setChild(True)
    cfe.setProperty('nre', nre)
    for k, v in kwargs.items():
        cfe.setProperty(k, v)
    cfe.execute()

    # Unpack the results
    eigenvalues = cfe.getProperty('Energies').value
    dim = len(eigenvalues)
    eigenvectors = _unpack_complex_matrix(cfe.getProperty('Eigenvectors').value, dim, dim)
    hamiltonian = _unpack_complex_matrix(cfe.getProperty('Hamiltonian').value, dim, dim)

    return eigenvalues, eigenvectors, hamiltonian

def fitengy(**kwargs):
    """ Uses the Newman-Ng algorithm to fit a set of crystal field parameters to a level scheme.

        blm = fitengy(Ion=ionname, sym=point_group, E=evec)
        blm = fitengy(Ion=ionname, E=evec, B=bvec)
        blm = fitengy(Ion=ionname, E=evec, B20=initB20, B40=initB40, ...)
        [B20, B40] = fitengy(IonNum=ionnumber, E=evec, B20=initB20, B40=initB40, OutputTuple=True)
        
        Note: This function only accepts keyword inputs.
        
        Inputs:
            ionname - name of the (tripositive) rare earth ion, e.g. 'Ce', 'Pr'.
            ionnumber - the number index of the rare earth ion: 
                   1=Ce 2=Pr 3=Nd 4=Pm 5=Sm 6=Eu 7=Gd 8=Tb 9=Dy 10=Ho 11=Er 12=Tm 13=Yb
            sym - a string with the Schoenflies symbol of the point group of the rare earth site
                   If bvec and sym are both given, bvec will take precedence (sym will be ignored)
            evec - a vector of the energy values to be fitted. Must equal 2J+1 for the selected ion.
            bvec - a vector of initial CF parameters in order: [B20 B21 B22 B40 B41 ... etc.]
                    zero values will not be fitted.
                    This vector can also a be dictionary instead {'B20':1, 'B40':2}
                    If no parameters are given but the symmetry is given random initial parameters
                    will be used, depending on the symmetry.
            B20 etc. - initial values of the CF parameters to be fitted. 

        Outputs:
            blm - a dictionary of the output crystal field parameters (default)
            [B20, etc] - a tuple of the output crystal field parameters (need to set OutputTuple flag)
    """

    # Some Error checking
    if 'Ion' not in kwargs.keys() and 'IonNum' not in kwargs.keys():
        raise NameError('You must specify the ion using either the ''Ion'' or ''IonNum'' keywords')
    if 'E' not in kwargs.keys():
        raise NameError('Input energy level scheme must be supplied as the ''E'' keyword input')
    E0 = np.array(sorted(kwargs['E']))# - np.mean(kwargs['E']))

    # Some definitions
    Blms = ['B20', 'B21', 'B22', 'B40', 'B41', 'B42', 'B43', 'B44', 'B60', 'B61', 'B62', 'B63', 'B64', 'B65', 'B66',
            'IB20', 'IB21', 'IB22', 'IB40', 'IB41', 'IB42', 'IB43', 'IB44', 'IB60', 'IB61', 'IB62', 'IB63', 'IB64', 'IB65', 'IB66']
    Ions = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb']

    if 'B' in kwargs.keys():
        bvec = kwargs.pop('B')
        if isinstance(bvec, dict):
            kwargs.update(bvec)
        else:
            for ii in range(len(bvec)):
                kwargs[Blms[ii]] = bvec[ii]

    if 'Ion' in kwargs.keys():
        nre = [ id for id,val in enumerate(Ions) if val==kwargs['Ion'] ][0] + 1
    else:
        nre = kwargs['IonNum']

    Jvals = [0, 5.0 / 2, 4, 9.0 / 2, 4, 5.0 / 2, 0, 7.0 / 2, 6, 15.0 / 2, 8, 15.0 / 2, 6, 7.0 / 2]
    J = Jvals[nre]
    #if len(E0) != int(2*J+1):
    #    raise RuntimeError(f'Expected {2*J+1} levels for Ion {Ions[nre]} but only got {len(E0)} elements in E0')

    if 'sym' in kwargs and len(set(kwargs.keys()).intersection(set(Blms))) == 0:
        # No parameters given, estimate using Monte Carlo sampling
        nz_pars = getSymmAllowedParam(kwargs['sym'])
        if J < 3:
            nz_pars = [v for v in nz_pars if 'B6' not in v]
        ebw = np.max(E0) - np.min(E0)
        ranges = split2range(Ion=Ions[nre], EnergySplitting=ebw, Parameters=nz_pars)
        # Estimate initial parameters using a Monte Carlo method
        kwargs.update({p:(np.random.rand()-0.5)*ranges[p] for p in nz_pars})
        kwargs['is_cubic'] = kwargs['sym'] in ['T', 'Td', 'Th', 'O', 'Oh']

    iscubic = kwargs.pop('iscubic', False)
    if iscubic:
        if 'B40' not in kwargs.keys() or 'B60' not in kwargs.keys():
            pass
        else:
            if 'B44' not in kwargs.keys():
                kwargs['B44'] = 5 * kwargs['B40']
            if 'B64' not in kwargs.keys():
                kwargs['B44'] = -21 * kwargs['B60']

    fitparind = []
    initBlm = {}
    for ind in range(len(Blms)):
        if Blms[ind] in kwargs.keys():
            fitparind.append(ind)
            initBlm[Blms[ind]] = kwargs[Blms[ind]]

    if not fitparind:
        raise NameError('You must specify at least one input Blm parameter')

    # Calculates the matrix elements <n|O_k^q|m>
    Omat = {}
    denom = {}
    for ind in fitparind:
        bdict = {Blms[ind]: 1}
        ee, vv, ham = CFEnergy(nre, **bdict)
        Omat[Blms[ind]] = np.mat(ham)
        denom[Blms[ind]] = np.trace( np.real( (Omat[Blms[ind]].H) * Omat[Blms[ind]] ))

    Ecalc, vv, ham = CFEnergy(nre, **initBlm)
    if len(E0) < len(Ecalc):
        #E = list(sorted(kwargs['E'])) + list(Ecalc[-(len(Ecalc)-len(E0)):]*(0.13*((100-num_iter)/100)+1) )
        # For each desired level, find nearest calculated level and substitute it for that
        Eref, Enear = (Ecalc, [])
        E = copy.deepcopy(Ecalc)
        for en in E0:
            Idif = np.argmin(np.abs(Eref - en))
            Enear.append(Eref[Idif])
            E[np.argmin(np.abs(Ecalc - Eref[Idif]))] = en
            Eref = np.delete(Eref, Idif)
        E0 = E - np.mean(E)
    else:
        E0 = E0 - np.mean(E0)

    lsqfit = 0
    Blm = initBlm
    div_count = 0
    for num_iter in range(100):
        if iscubic:
            Blm['B44'] = 5 * Blm['B40']
            Blm['B64'] = -21 * Blm['B60']
        Ecalc, vv, ham = CFEnergy(nre, **Blm)
        V = np.mat(vv)
        Ecalc = Ecalc - np.mean(Ecalc)
        newlsqfit = np.sum(np.power(Ecalc-E0,2))
        if np.fabs(lsqfit - newlsqfit)<1.e-7:
            break
        if newlsqfit > lsqfit:
            div_count += 1
        if div_count > 10:
            warnings.warn('Fit is diverging')
            break
        lsqfit = newlsqfit
        for ind in fitparind:
            # Calculates the numerator = sum_n En <j|Okq|i>_nn
            numer = np.dot( np.real( np.diag( V.H * Omat[Blms[ind]] * V ) ), E0 )
            # Calculates the new Blm parameter
            Blm[Blms[ind]] = numer / denom[Blms[ind]]

    if 'OutputTuple' in kwargs.keys() and kwargs['OutputTuple']:
        retval = []
        for ind in fitparind:
            retval.append(Blm[Blms[ind]])
        return tuple(retval)
    else:
        return Blm

BLMS = ['B20', 'B21', 'B22', 'B40', 'B41', 'B42', 'B43', 'B44', 'B60', 'B61', 'B62', 'B63', 'B64', 'B65', 'B66',
        'IB21', 'IB22', 'IB41', 'IB42', 'IB43', 'IB44', 'IB61', 'IB62', 'IB63', 'IB64', 'IB65', 'IB66']
SIGMAFAC = 2 * np.sqrt(2 * np.log(2))


class CEFData:

    def get_data_from_workspace(self, InputWorkspace):
        ws = s_api.mtd[InputWorkspace] if isinstance(InputWorkspace, str) else InputWorkspace
        x = np.squeeze(ws.extractX())
        assert len(x.shape) == 1, "Error: input workspace must be 1D"
        y = np.squeeze(ws.extractY())
        e = np.squeeze(ws.extractE())
        if len(x) == (len(y) + 1):
            x = (x[:-1] + x[1:]) / 2.0
        assert len(x) == len(y), "Error: x- and y- dimensions are not consistent"
        return x, y, e

    def __init__(self, InputWorkspace):
        if hasattr(InputWorkspace, '__len__'):
            self.nset = len(InputWorkspace)
            self.data = [self.get_data_from_workspace(ws) for ws in InputWorkspace]
        else:
            self.nset = 1
            self.data = [self.get_data_from_workspace(ws)]

    def xye(self, index=0):
        return self.data[index][0], self.data[index][1], self.data[index][2]

    def max_x(self):
        return np.max([np.max(self.data[ii][0]) for ii in range(self.nset)])


def gauss(x, cen, area, fwhm):
    x = np.array(x)
    flgt = 4 * np.log(2)
    fac = np.sqrt(flgt / np.pi)
    return (area / fwhm * fac) * np.exp(-flgt * ((x - cen) / fwhm)**2)


def lorz(x, cen, area, fwhm):
    x = np.array(x)
    return (area / np.pi * (fwhm / 2)) / ( (x - cen)**2 + (fwhm / 2)**2 )


def voigt(x, cen, area, fwhm, frac=0.5):
    x = np.array(x)
    flgt = 4 * np.log(2)
    return (area/fwhm) / (frac*np.pi/2 + (1-frac)*np.sqrt(np.pi / flgt)) \
        * (frac/(1 + 4*((x - cen)/fwhm)**2) + (1-frac)*np.exp(-flgt*((x - cen)/fwhm)**2))


def parse_cef_func(func, is_voigt=False):
    cffun = func
    if isinstance(func, mantid.api.CompositeFunction):
        for fn in func:
            try:
                tolint = fn.getAttributeValue('ToleranceIntensity')
                cffun = fn
            except:
                pass
            else:
                break
    else:
        tolint = func.getAttributeValue('ToleranceIntensity')
    fstr, ties, constraints = (str(cffun), '()', '()')
    if ',ties=' in fstr:
        fstr, ties = tuple(fstr.split(',ties='))
    if ',constraints=' in fstr:
        fstr, constraints = tuple(fstr.split(',constraints='))
    specs = re.findall('sp[0-9]*\.IntensityScaling', fstr)
    if specs:              # Multi datasets
        spe_intscal = [float(ff.group(1)) for ff in re.finditer('sp[0-9]*\.IntensityScaling=([0-9\.e\-]*)', fstr)]
        specs = [fs.split('.IntensityScaling')[0]+'.' for fs in specs]
        spe_intname = [f'sp{ii}.IntensityScaling' for ii in range(len(spe_intscal))]
    else:
        specs = re.findall('IntensityScaling[0-9]+=', fstr)
        if specs:
            spe_intscal = [float(ff.group(1)) for ff in re.finditer('IntensityScaling[0-9]+=([0-9\.e\-]*)', fstr)]
            specs = [f'f{n}.' for n in range(len(specs))]
            spe_intname = [f'IntensityScaling{ii}' for ii in range(len(spe_intscal))]
        else:
            try:
                spe_intscal = [float(re.search(',IntensityScaling=([0-9\.e\-]*)', fstr).group(1))]
            except AttributeError:    # CrystalFieldMultiSpectrum (but single site) does not have this parameter
                spe_intscal = [1.0]
            specs = ['']
            spe_intname = ['IntensityScaling']
    if 'Temperatures=' in fstr:
        tt = re.search('Temperatures=\(([\,0-9\.]*)\)', fstr).group(1).split(',')
    else:
        tt = [re.search('Temperature=([0-9]*)', fstr).group(1)]
    sites = re.findall('ion[0-9]*\.B20=', fstr)
    if sites:
        sites = [fs.split('.B20=')[0]+'.' for fs in sites]
        ions = re.search('Ions="([\ A-z\,]*)"', fstr).group(1).split(',')
        syms = re.search('Symmetries="([\ A-z0-9\,]*)"', fstr).group(1).split(',')
        ion_intscal = [float(ff.group(1)) for ff in re.finditer('ion[0-9]*\.IntensityScaling=([0-9\.e\-]*)', fstr)]
    else:
        sites = ['']
        ions = [re.search('Ion=([A-z]*)', fstr).group(1)]
        syms = [re.search('Symmetry=([A-z0-9]*)', fstr).group(1)]
        ion_intscal = [1.0]
    blms, cfob, pks, w0 = ([], [], [], {})
    if not is_voigt:
        is_voigt = re.search('PeakShape=([A-z]*),', fstr).group(1) == 'PseudoVoigt'
    for ii, ss in enumerate(sites):
        blms.append({pn:cffun[f'{ss}{pn}'] for pn in BLMS if f'{ss}{pn}=' in fstr})
        # Overwrite values with ties if they exists
        blms[-1].update({pn:float(re.search(f'{ss}{pn}=([\-0-9\.e]*)', ties).group(1)) for pn in BLMS if f'{ss}{pn}=' in ties})
        cfob.append([])
        pks.append([])
        for jj, sp in enumerate(specs):
            cfob[-1].append(CrystalField(ions[ii], syms[ii], Temperature=float(tt[jj]), **blms[-1]))
            pkn = re.findall(f'{ss}{sp}([A-z0-9]*)\.PeakCentre=', fstr)
            #pks[-1].append({pk:{ff.group(1):float(ff.group(2)) for ff in re.finditer(f'{ss}{sp}{pk}\.([A-z]*)=([\-0-9\.]*)', fstr)} for pk in pkn})
            pks[-1].append({})
            pk0 = cfob[-1][-1].getPeakList()
            for pk in [pkv for pkv in pkn if pkv]:
                pkdic = {ff.group(1):float(ff.group(2)) for ff in re.finditer(f'{ss}{sp}{pk}\.([A-z]*)=([\-0-9\.e]*)', fstr)}
                idx = np.argmin(np.abs(pk0[0] - pkdic['PeakCentre']))
                if 'Amplitude' not in pkdic:
                    pkdic['Amplitude'] = pk0[1,idx] * spe_intscal[jj] * ion_intscal[ii]
                elif pkdic['Amplitude'] >= tolint:
                    if abs(pkdic['Amplitude'] - pk0[1,idx] * spe_intscal[jj] * ion_intscal[ii]) > 0.2:
                        print(f"Warning: inconsistent peak {pkdic['Amplitude']}, {pk0[1,idx] * spe_intscal[jj] * ion_intscal[ii]}")
                        continue
                        #raise RuntimeError("Error: inconsistent peak intensities")
                if 'Sigma' in pkdic:
                    pkdic['FWHM'] = SIGMAFAC * pkdic['Sigma']
                    w0[f'{ss}{sp}{pk}.Sigma'] = pkdic['Sigma']
                else:
                    w0[f'{ss}{sp}{pk}.FWHM'] = pkdic['FWHM']
                if is_voigt:
                    if 'Mixing' not in pkdic:
                        pkdic['Mixing'] = 0.5
                    w0[f'{ss}{sp}{pk}.Mixing'] = pkdic['Mixing']
                if abs(pk0[0, idx] - pkdic['PeakCentre']) < 0.1 and pk0[1, idx] > tolint:
                    pks[-1][-1][f'{ss}{sp}{pk}'] = pkdic
    return blms, cfob, pks, [spe_intscal, spe_intname], w0


def get_peaklist_from_dic(pkdic, intscal, bkg, maxE=9e99):
    nset = len(pkdic[0])
    peaklist, p0, pnames = ([], [], [])
    for ii in range(nset):
        celllist, cellp0, cellpnames, firstpeak = ([], [], [], True)
        for jj in range(len(pkdic)):
            for pkn, val in pkdic[jj][ii].items():
                if val['PeakCentre'] < maxE or firstpeak:
                    celllist.append([val['PeakCentre'], val['Amplitude']])
                    cellp0.append(val['FWHM'])
                    cellpnames.append(pkn)
                    firstpeak = False
        peaklist.append(np.array(celllist).T)
        p0.append(cellp0 + [intscal[ii], bkg])
        pnames.append(cellpnames)
    return peaklist, p0, pnames


def fit_widths(fitobj, retres=False, is_voigt=False, **kwargs):
    cfpars, cfobjs, peaks, intscal, _ = parse_cef_func(fitobj.model.function, is_voigt)
    bkg = 0
    if hasattr(fitobj.model, '_background') and fitobj.model.background is not None:
        bkgobj = fitobj.model.background
        if isinstance(bkgobj, list):  # Is a single-site multi spectrum
            bkgobj = bkgobj[0]
        try:
            bkg = bkgobj.background.function.getParameterValue('f0.A0')
        except:
            if 'LinearBackground' in str(bkgobj.background.function):
                bkg = float(re.search('LinearBackground,A0=([0-9\.e\-]*),A1', str(bkgobj.background.function)).group(1))
    if not is_voigt and fitobj.model.PeakShape == 'PseudoVoigt':
        is_voigt = True
    if is_voigt:
        peakfun = voigt
    elif fitobj.model.PeakShape == 'Lorentzian':
        peakfun = lorz
    elif fitobj.model.PeakShape == 'Gaussian':
        peakfun = gauss
        fac = np.sqrt(4 * np.log(2)) / np.pi
    else:
        raise RuntimeError('Only Lorentzian, Gaussian or PseudoVoigt peak shapes are supported')
    def minfun(pkl, pp, x, y, e, rr=False):
        if len(pkl) == 0:
            if rr:
                return np.sum((2*y)**2 / (e**2)), np.abs(2*y)
            return np.sum((2*y)**2 / (e**2))
        npk = pkl.shape[1]
        yf = np.zeros(len(x))
        #assert len(pp) == (npk + 2), "Error: length of widths input is not consistent with peak list"
        if is_voigt:
            for ii in range(npk):
                yf += peakfun(x, pkl[0, ii], pkl[1, ii], pp[ii], pp[npk+ii])
        else:
            for ii in range(npk):
                yf += peakfun(x, pkl[0, ii], pkl[1, ii], pp[ii])
        yf = yf * pp[-2] + pp[-1]
        if rr:
            return np.sum((y - yf)**2 / (e**2)), np.abs(y - yf)
        return np.sum((y - yf)**2 / (e**2))
    data = CEFData(fitobj._input_workspace)
    maxE = data.max_x() * 2
    peaklist, p0, pnam = get_peaklist_from_dic(peaks, intscal[0], bkg, maxE)
    if is_voigt:
        for ii in range(data.nset):
            npk = peaklist[ii].shape[1]
            p0[ii] = p0[ii][:npk] + [0.5]*npk + p0[ii][-2:]
    assert data.nset == len(p0), "Incorrect number of datasets compared to model"
    maxfwhm = kwargs.pop('maxfwhm', [-1]*data.nset)
    chi2 = 0.0
    resi, pfit = ([], [])
    for ii in range(data.nset):
        bnd = [(0.2*maxfwhm[ii], maxfwhm[ii]) if maxfwhm[ii] > 0 else (0.2*pv, 3*pv) for pv in p0[ii]]
        p0[ii][-2] = abs(p0[ii][-2])
        p0[ii][-1] = abs(p0[ii][-1])
        bnd[-2] = (0, p0[ii][-2]*3)  # Scale factor
        bnd[-1] = (0, p0[ii][-1]*3)  # Background
        if is_voigt:
            for jj in range(npk, len(p0[ii])-1):
                bnd[jj] = (0.0, 1.0)
        res = scipy.optimize.minimize(lambda p: minfun(peaklist[ii], p, *data.xye(ii)), p0[ii], bounds=bnd, **kwargs)
        #if not res.success:
        #    raise RuntimeError(f'Fitting widths failed with error: "{res.message}"')
        if retres:
            x, y, e = data.xye(ii)
            c2, resi0 = minfun(peaklist[ii], res.x, x, y, e, rr=True)
            chi2 += c2
            resi += resi0.tolist()
        else:
            chi2 += minfun(peaklist[ii], res.x, *data.xye(ii))
        #print(res)
        #print(res.x)
        # Updates the fit model
        if fitobj.model.PeakShape == 'Gaussian':
            for jj, pkn in enumerate(pnam[ii]):
                fitobj.model[f'{pkn}.Sigma'] = res.x[jj] / SIGMAFAC
                fitobj.model[f'{pkn}.Height'] = (peaklist[ii][1, jj] / res.x[jj] * fac)
        else:
            for jj, pkn in enumerate(pnam[ii]):
                fitobj.model[f'{pkn}.FWHM'] = res.x[jj]
        fitobj.model[intscal[1][ii]] = res.x[-2]
        #fitobj.model
        if is_voigt:
            pfit.append(res.x)
    if is_voigt:
        evalfit_voigt(fitobj, peaklist, pfit)
    if retres:
        return chi2, np.array(resi)
    return chi2

def fit_cef(fitobj, **kwargs):
    cfpars, cfobjs, peaks, intscal, origwidths = parse_cef_func(fitobj.model.function)
    p0, pnam, bnd = ([], [], [])
    if len(cfpars) > 1:
        for ii in range(len(cfpars)):
            ranges = split2range(Ion=cfobjs[ii][0].Ion, EnergySplitting=np.max(cfobjs[ii][0].getPeakList()[0]), Parameters=list(cfpars[ii].keys()))
            for ky, vl in cfpars[ii].items():
                if abs(vl) > 0:
                    p0.append(vl)
                    pnam.append(f'ion{ii}.{ky}')
                    bv = ranges[ky.replace('I','')]
                    if abs(vl) > bv:
                        bv = abs(vl) * 3
                    bnd.append((-bv, bv))
    else:
        ranges = split2range(Ion=cfobjs[0][0].Ion, EnergySplitting=np.max(cfobjs[0][0].getPeakList()[0]), Parameters=list(cfpars[0].keys()))
        for ky, vl in cfpars[0].items():
            if abs(vl) > 0:
                p0.append(vl)
                pnam.append(ky)
                bv = ranges[ky.replace('I','')]
                if abs(vl) > bv:
                    bv = abs(vl) * 3
                bnd.append((-bv, bv))
    widths_kw = kwargs.pop('widths_kwargs', kwargs)
    chi2v = []
    def minfun(p):
        for ky, vl in zip(pnam, p):
            fitobj.model[ky] = vl
        for ky, vl in origwidths.items():
            fitobj.model[ky] = vl
        chi2 = fit_widths(fitobj, **widths_kw)
        if len(chi2v) > 2 and chi2 < np.min(chi2v):
            mantid.simpleapi.CreateWorkspace(range(len(p)), p, OutputWorkspace='bestpars')
        chi2v.append(chi2)
        try:
            fitobj.fit()
        except ValueError:
            pass
        if len(chi2v) < 2:
            chi2v.append(chi2)
        mantid.simpleapi.CreateWorkspace(range(len(chi2v)), chi2v, OutputWorkspace='chi2')
        return chi2
    orig_maxiter = fitobj._fit_properties.pop('MaxIterations', None)
    fitobj._fit_properties['MaxIterations'] = 0
    kws = copy.deepcopy(kwargs)
    kws.pop('maxfwhm', None)
    res = scipy.optimize.minimize(minfun, p0, bounds=bnd, **kws)
    #res = scipy.optimize.differential_evolution(minfun, bounds=bnd, x0=p0)
    if orig_maxiter is not None:
        fitobj._fit_properties['MaxIterations'] = orig_maxiter
    else:
        fitobj._fit_properties.pop('MaxIterations')
    # Update with best fit parameters
    minfun(res.x)
    return res


def evalfit_voigt(fitobj, peaklist=None, pfit=None):
    #assert fitobj.model.PeakShape == 'PseudoVoigt', 'This function is only for PseudoVoigt peak shapes'
    if pfit is None:
        fit_widths(fitobj, is_voigt=True)
        return
    data = CEFData(fitobj._input_workspace)
    if peaklist is None:
        if hasattr(fitobj.model, '_background') and fitobj.model.background is not None:
            bkgobj = fitobj.model.background
            if isinstance(bkgobj, list):  # Is a single-site multi spectrum
                bkgobj = bkgobj[0]
            try:
                bkg = bkgobj.background.function.getParameterValue('f0.A0')
            except:
                if 'LinearBackground' in str(bkgobj.background.function):
                    bkg = float(re.search('LinearBackground,A0=([0-9\.e\-]*),A1', str(bkgobj.background.function)).group(1))
                else:
                    bkg = 0.0
        bkg = [bkg]*data.nset
        _, _, peaks, intscal, _ = parse_cef_func(fitobj.model.function)
        peaklist, p0, pnam = get_peaklist_from_dic(peaks, intscal[0], bkg)
        scalfac = [intscal[0][ii] for ii in range(len(intscal))]
    elif pfit is not None:   # peaklist and pfit given
        scalfac = [pfit[ii][-2] for ii in range(data.nset)]
        bkg = [pfit[ii][-1] for ii in range(data.nset)]
        mixing = [pfit[ii][peaklist[ii].shape[1]:-2] for ii in range(data.nset)]
    else:                    # only peaklist given
        scalfac, bkg, mixing = ([1.0]*data.nset, [0.0]*data.nset, [[0.5]*peaklist[ii].shape[1] for ii in range(data.nset)])
    #nsites = len(peaks)
    assert data.nset == len(peaklist), 'Inconsistent number of datasets in fit object'
    bn = fitobj._output_workspace_base_name
    for ii in range(data.nset):
        x, y, e = data.xye(ii)
        yf, z = np.zeros(len(x)), np.zeros(len(x))
        npk = peaklist[ii].shape[1]
        for jj in range(npk):
            yf += voigt(x, peaklist[ii][0, jj], peaklist[ii][1, jj], pfit[ii][jj], mixing[ii][jj])
        yf = yf*scalfac[ii] + bkg[ii]
        wsn = bn + '_Workspace' + f'_{ii}' if data.nset > 1 else ''
        mantid.simpleapi.CreateWorkspace(x, np.array([y, yf, y-yf]), np.array([e, z, z]), NSpec=3, OutputWorkspace=wsn)
    if data.nset > 1:
        mantid.simpleapi.GroupWorkspaces([f'{bn}_Workspace_{ii}' for ii in range(data.nset)], OutputWorkspace=f'{bn}_Workspaces')


def fit_en(fitobj, enlist, eval_only=None, is_voigt=False, **kwargs):
    if is_voigt:
        origshape = fitobj.model.PeakShape
        fitobj.model.PeakShape = 'Lorentzian'
        fstr = str(fitobj.model.function)
    cfpars, cfobjs, peaks, intscal, origwidths = parse_cef_func(fitobj.model.function)
    p0, pnam, bnd = ([], [], [])
    if len(cfpars) > 1:
        for ii in range(len(cfpars)):
            ranges = split2range(Ion=cfobjs[ii][0].Ion, EnergySplitting=np.max(cfobjs[ii][0].getPeakList()[0]), Parameters=list(cfpars[ii].keys()))
            for ky, vl in cfpars[ii].items():
                if abs(vl) > 0:
                    p0.append(vl)
                    pnam.append(f'ion{ii}.{ky}')
                    bv = ranges[ky.replace('I','')]
                    if abs(vl) > bv:
                        bv = abs(vl) * 3
                    bnd.append((-bv, bv))
    else:
        ranges = split2range(Ion=cfobjs[0][0].Ion, EnergySplitting=np.max(cfobjs[0][0].getPeakList()[0]), Parameters=list(cfpars[0].keys()))
        for ky, vl in cfpars[0].items():
            if abs(vl) > 0:
                p0.append(vl)
                pnam.append(ky)
                bv = ranges[ky.replace('I','')]
                if abs(vl) > bv:
                    bv = abs(vl) * 3
                bnd.append((-bv, bv))
    widths_kw = kwargs.pop('widths_kwargs', kwargs)
    fit_alg = kwargs.pop('fit_alg', 'local')
    extra_chi2_fun = kwargs.pop('extra_chi2_fun', None)
    chi2v = []
    ions = [cfobjs[ii][0].Ion for ii in range(len(cfpars))]
    is_voigt = is_voigt or fitobj.model.PeakShape == 'PseudoVoigt'
    def minfun(p, retres=False):
        blms = []
        if len(cfpars) > 1:
            for ii in range(len(cfpars)):
                b0 = {ky.split(f'ion{ii}.')[1]:vl for ky, vl in zip(pnam, p) if f'ion{ii}.' in ky}
                for ky, vl in fitengy(Ion=ions[ii], E=enlist[ii], B=b0).items():
                    fitobj.model[f'ion{ii}.{ky}'] = vl
                fitobj.model[intscal[1][ii]] = intscal[0][ii]
                blms.append(b0)
        else:
            b0 = {ky:vl for ky, vl in zip(pnam, p)}
            for ky, vl in fitengy(Ion=cfobjs[0][0].Ion, E=enlist, B=b0).items():
                fitobj.model[ky] = vl
            blms.append(b0)
        for ky, vl in origwidths.items():
            fitobj.model[ky] = vl
        if retres:
            chi2, resi = fit_widths(fitobj, retres=True, is_voigt=is_voigt, **widths_kw)
        else:
            chi2 = fit_widths(fitobj, is_voigt=is_voigt, **widths_kw)
        if extra_chi2_fun is not None:
            chi2 += extra_chi2_fun(blms)
        if len(chi2v) > 2 and chi2 < np.min(chi2v):
            mantid.simpleapi.CreateWorkspace(range(len(p)), p, OutputWorkspace='bestpars')
            genpp(fitobj)
        chi2v.append(chi2)
        if not is_voigt:
            try:
                fitobj.fit()
            except ValueError:
                pass
        if len(chi2v) < 2:
            chi2v.append(chi2)
        mantid.simpleapi.CreateWorkspace(range(len(chi2v)), chi2v, OutputWorkspace='chi2')
        if retres:
            return resi
        return chi2
    if eval_only is not None:
        return minfun(eval_only)
    orig_maxiter = fitobj._fit_properties.pop('MaxIterations', None)
    fitobj._fit_properties['MaxIterations'] = 0
    kws = copy.deepcopy(kwargs)
    kws.pop('maxfwhm', None)
    if fit_alg == 'gofit':
        if gofit is None:
            raise RuntimeError('gofit is not installed on this system')
        class gfres:
            def __init__(self, inp):
                self.x = inp[0]
                self.stat = inp[1]
        dt = CEFData(fitobj._input_workspace)
        mxi = orig_maxiter if orig_maxiter is not None else 100
        opt = kws.pop('options', {'maxiter':mxi, 'samples':10})
        m = np.sum([len(d[0]) for d in dt.data])
        xl, xu = np.array([bn[0] for bn in bnd]), np.array([bn[1] for bn in bnd])
        res = gfres(gofit.multistart(m, len(p0), xl, xu, lambda px: minfun(px, retres=True),
                                     samples=opt.pop('samples', 10), maxit=opt.pop('maxiter', mxi)))
    elif fit_alg == 'global':
        #res = scipy.optimize.differential_evolution(minfun, bounds=bnd, x0=p0)
        #res = scipy.optimize.dual_annealing(minfun, bounds=bnd, x0=p0)
        algo_name = kwargs.pop('algorithm', 'differential_evolution')
        assert algo_name != 'brute', "Brute force algorithm is not supported"
        try:
            algo = getattr(scipy.optimize, algo_name)
        except AttributeError:
            raise RuntimeError('Unknown global algorithm: ' + algo_name)
        else:
            if algo_name == 'basinhopping':
                res = algo(minfun, x0=p0)
            elif algo_name == 'shgo' or algo_name == 'direct':
                res = algo(minfun, bounds=bnd)
            else:
                res = algo(minfun, bounds=bnd, x0=p0)
    else:
        res = scipy.optimize.minimize(minfun, p0, bounds=bnd, **kws)
    if orig_maxiter is not None:
        fitobj._fit_properties['MaxIterations'] = orig_maxiter
    else:
        fitobj._fit_properties.pop('MaxIterations')
    if is_voigt:
        fitobj.model.PeakShape = origshape
    # Update with best fit parameters
    minfun(res.x)
    return res


def genpp(fitobj):
    from mantid.simpleapi import CreateWorkspace, RenameWorkspace, CloneWorkspace, Plus, Divide, CreateSingleValuedWorkspace
    cfpars, cfobjs, peaks, intscal, origwidths = parse_cef_func(fitobj.model.function)
    tt = np.arange(1,300,1)
    hh = np.linspace(0,10,100)
    print(cfobjs)
    if 1:#len(cfobjs) > 1:
        invchi_x, invchi_y, invchi_z, invchi_p = ([], [], [], [])
        mag_x, mag_y, mag_z, mag_p = ([], [], [], [])
        for ii in range(len(cfobjs)):
            invchi_x.append(CreateWorkspace(*cfobjs[ii][0].getSusceptibility(tt, Hdir=[1, 0, 0], Inverse=True, Unit='cgs'), OutputWorkspace=f'invchi_{ii}_x'))
            invchi_y.append(CreateWorkspace(*cfobjs[ii][0].getSusceptibility(tt, Hdir=[0, 1, 0], Inverse=True, Unit='cgs'), OutputWorkspace=f'invchi_{ii}_y'))
            invchi_z.append(CreateWorkspace(*cfobjs[ii][0].getSusceptibility(tt, Hdir=[0, 0, 1], Inverse=True, Unit='cgs'), OutputWorkspace=f'invchi_{ii}_z'))
            tmp_ws = (invchi_x[-1] + invchi_y[-1] + invchi_z[-1]) / 3
            tmp_ws = RenameWorkspace(tmp_ws, OutputWorkspace=f'invchi_{ii}_p')
            invchi_p.append(tmp_ws)
            mag_x.append(CreateWorkspace(*cfobjs[ii][0].getMagneticMoment(T=1.5, Hmag=hh, Hdir=[1, 0, 0], Unit='bohr'), OutputWorkspace=f'mag_{ii}_x'))
            mag_y.append(CreateWorkspace(*cfobjs[ii][0].getMagneticMoment(T=1.5, Hmag=hh, Hdir=[0, 1, 0], Unit='bohr'), OutputWorkspace=f'mag_{ii}_y'))
            mag_z.append(CreateWorkspace(*cfobjs[ii][0].getMagneticMoment(T=1.5, Hmag=hh, Hdir=[0, 0, 1], Unit='bohr'), OutputWorkspace=f'mag_{ii}_z'))
            tmp_ws = (mag_x[-1] + mag_y[-1] + mag_z[-1]) / 3
            tmp_ws = RenameWorkspace(tmp_ws, OutputWorkspace=f'mag_{ii}_p')
            mag_p.append(tmp_ws)
        wss = [[invchi_x, invchi_y, invchi_z, invchi_p], [mag_x, mag_y, mag_z, mag_p]]
        tmp_denom = CreateSingleValuedWorkspace(len(cfobjs))
        for kk, prf in enumerate(['invchi', 'mag']):
            for ii, suf in enumerate(['x', 'y', 'z', 'p']):
                wsn = f'{prf}_{suf}'
                tmp_ws = CloneWorkspace(wss[kk][ii][0], OutputWorkspace=wsn)
                for jj in range(1, len(cfobjs)):
                    tmp_ws = Plus(tmp_ws, wss[kk][ii][jj], OutputWorkspace=wsn)
                tmp_ws = Divide(tmp_ws, tmp_denom, OutputWorkspace=wsn)


def printpars(fitobj):
    cfpars, cfobjs, peaks, intscal, origwidths = parse_cef_func(fitobj.model.function)
    if len(cfobjs) > 1:
        for ii in range(len(cfobjs)):
            for bln in BLMS:
                if bln in cfpars[ii] and abs(cfpars[ii][bln]) > 1e-10:
                    print(f'ion{ii}.{bln} = {cfpars[ii][bln]:0.5g}')
    else:
        for bln in BLMS:
            if bln in cfpars[0] and abs(cfpars[0][bln]) > 1e-6:
                print(f'{bln} = {cfpars[0][bln]:0.5g}')
