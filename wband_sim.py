# CODE TO GET ABSOLUTE MAGNITUDES IN CUSTOM FILTER FOR LIST OF STARS WITH KNOWN SPECTRAL TYPES

import sys
import glob
import scipy as sc
import numpy as np
import scipy.io as scio
from astropy.io import fits
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
import fnmatch
import dereddening


def idl_tabulate(x, f, p=5) :
    def newton_cotes(x, f) :
        if x.shape[0] < 2 :
            return 0
        rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
        weights = integrate.newton_cotes(rn)[0]
        return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
    ret = 0
    for idx in range(0, x.shape[0], p - 1) :
        ret += newton_cotes(x[idx:idx + p], f[idx:idx + p])
    return ret

def new_flux(wl,fl,fwl,ffl,onoff,trans):

        wl = np.asarray(wl)
        fl = np.asarray(fl)
        abs_fl = np.asarray([abs(i) for i in fl])

        good=np.where(abs_fl > 0.0)
        bad=np.where(abs_fl <= 0.0)
        if len(bad[0]) > 0:
            fl[bad] = np.nan
    #remove nan flux points and corresponding wavelength points
        wl = wl[np.where(~np.isnan(fl))]
        fl = fl[np.where(~np.isnan(fl))]
        wl = wl[np.where(np.isfinite(fl))]
        fl = fl[np.where(np.isfinite(fl))]

        atmo_wl,atmo_trans = [],[]
        a=open('mktrans_zm_30_15.dat','r').readlines()
        for line in a:
            data = line.split()
            atmo_trans.append(float(data[1]))
            atmo_wl.append(float(data[0]))

        atmo_trans = np.asarray(atmo_trans)
        atmo_wl = np.asarray(atmo_wl)
        atmo_trans /= max(atmo_trans)

        if trans:

            filter_func = interpolate.interp1d(fwl,ffl,bounds_error=False,fill_value=0.0)
            filt = filter_func(wl)

            func2 = interpolate.interp1d(atmo_wl,atmo_trans,bounds_error=False, fill_value=0.0)
            filtatmo = func2(wl)
            fflux = idl_tabulate(wl, fl*wl*filtatmo*filt)/idl_tabulate(wl, wl*filtatmo*filt)

        else:

            filt = np.zeros(len(wl))
            thru = np.where(np.logical_and(np.greater_equal(wl,onoff[0]),np.less_equal(wl,onoff[1])))
            filt[thru]=100

            func2 = interpolate.interp1d(atmo_wl,atmo_trans,bounds_error=False, fill_value=0.0)
            filtatmo = func2(wl)
            
            fflux = idl_tabulate(wl, fl*wl*filtatmo*filt)/idl_tabulate(wl, wl*filtatmo*filt)

        return fflux

def calc_mags(spt,av,filetype):

    #READ IN H AND J TRANSMISSION CURVES
        transfileJ = open('CFHT_J_TRANSMISSION.dat','r').readlines()    
        transfileH = open('CFHT_H_TRANSMISSION.dat','r').readlines()
        transfileK = open('CFHT_Ks_TRANSMISSION.dat','r').readlines()
        transfileW = open('CFHT_W_TRANSMISSION.dat','r').readlines()

        filt_wlJ,filt_transJ,filt_wlH,filt_transH,filt_wlK,filt_transK,filt_wlW,filt_transW = [],[],[],[],[],[],[],[]

        for line in transfileJ:
            data = line.split()
            filt_wlJ.append(float(data[0])/1000.)
            filt_transJ.append(float(data[1]))

        for line in transfileH:
            data = line.split()
            filt_wlH.append(float(data[0])/1000.)
            filt_transH.append(float(data[1]))

        for line in transfileW:
            data = line.split(',')
            filt_wlW.append(float(data[0])/1000.)
            filt_transW.append(float(data[1]))

        for line in transfileK:
            data = line.split()
            filt_wlK.append(float(data[0])/1000.)
            filt_transK.append(float(data[1]))

        #filenames1 = sorted(sys.argv[2:])        

        if 'bs' in filetype or 'tl' in filetype:
            filenames = glob.glob('std2/*fits')        

        elif 'l17' in filetype:
            filenames=glob.glob('l17/*fits')

        # READ IN VEGA A0 STANDARD
        b=scio.readsav('alpha_lyr_stis_005.sav',python_dict=True)
        c=list(b.values())
        vega = [[0 for j in range(len(c[0]))] for i in range(2)]
        indwin = np.where(np.logical_and(np.greater_equal(c[0],5551),np.less_equal(c[0],5561)))
        scale=3.46e-9/np.mean(c[1][indwin])
                
        for i in range(len(vega[0])):
            vega[0][i] = c[0][i]/10000
            vega[1][i] = c[1][i]*scale*10

        # FIND MATCHING SPECTRAL TYPE FOR BACKGROUND MODEL FILES
        if 'bs' in filetype or 'tl' in filetype:
            possfiles = []
            sptfiletest = str('*' + str(spt) + '*.fits')
            for i in range(len(filenames)):
                if fnmatch.fnmatch(str(filenames[i]),sptfiletest):
                    possfiles.append(filenames[i])

            # Pick a random option from the standards available

            num = len(possfiles)
            choice = np.random.randint(0,num) 
            chosenfile = possfiles[choice]

            stan = fits.open(str(chosenfile))
            spectra = stan[0].data
            wl = spectra[0]
            fl = spectra[1]
            head = stan[0].header
            happ = head['H']

            obj = [[0 for j in range(len(wl))] for i in range(2)]
            obj[0][:] = wl
            obj[1][:] = fl
 

        # FIND MATCHING SPECTRAL TYPE FOR LUHMAN 17 STANDARDS
        elif 'l17' in filetype:

            sptfiletest = str('*' + str(spt) + '*.fits')

            for i in range(len(filenames)):
                if fnmatch.fnmatch(str(filenames[i]),sptfiletest):
                    chosenfile = filenames[i]

            stan = fits.open(str(chosenfile))
            header = stan[0].header
            fl=stan[0].data
            if header['WCSDIM'] ==1:
                wl = np.arange(header['CRVAL1']/10000.0,header['CRVAL1']/10000.0+(len(fl)*round(header['CDELT1'])/10000.0),round(header['CDELT1'])/10000.0)
            else:
                wl = np.arange(7000.0/10000.0,7000.0/10000.0+(len(fl)*(20.0/10000.0)),20.0/10000.0)

            obj = [[0 for j in range(len(wl))] for i in range(2)]
            obj[0][:] = wl
            obj[1][:] = fl

        elif 'realfile' in filetype:

            stan = fits.open('l17/'+str(spt))
            header = stan[0].header
            fl=stan[0].data
            if header['WCSDIM'] ==1:
                wl = np.arange(header['CRVAL1']/10000.0,header['CRVAL1']/10000.0+(len(fl)*round(header['CDELT1'])/10000.0),round(header['CDELT1'])/10000.0)
            else:
                wl = np.arange(7000.0/10000.0,7000.0/10000.0+(len(fl)*(20.0/10000.0)),20.0/10000.0)

            obj = [[0 for j in range(len(wl))] for i in range(2)]
            obj[0][:] = wl
            obj[1][:] = fl
        if av != 0:
            obj[1][:] = dereddening.dered(obj[0],obj[1],av)

        #plt.figure()
        #plt.plot(filt_wlJ,filt_transJ,c='k')
       # plt.plot(filt_wlH,filt_transH,c='r')
       # plt.plot(filt_wlW,filt_transW,c='b')
       ## plt.plot(filt_wlK,filt_transK,c='g')
       # plt.show()

        # CALCULATE 'MAGNITUDES' FOR CFHT PHOTOMETRIC BANDS 
        broadmagJ = -2.5*np.log10(new_flux(obj[0],obj[1],filt_wlJ,filt_transJ,0.0,True)/new_flux(vega[0],vega[1],filt_wlJ,filt_transJ,0.0,True))
        broadmagH = -2.5*np.log10(new_flux(obj[0],obj[1],filt_wlH,filt_transH,0.0,True)/new_flux(vega[0],vega[1],filt_wlH,filt_transH,0.0,True))
        broadmagK = -2.5*np.log10(new_flux(obj[0],obj[1],filt_wlK,filt_transK,0.0,True)/new_flux(vega[0],vega[1],filt_wlK,filt_transK,0.0,True))
        cusmagW = -2.5*np.log10(new_flux(obj[0],obj[1],filt_wlW,filt_transW,0.0,True)/new_flux(vega[0],vega[1],filt_wlW,filt_transW,0.0,True))

        #onoff1 = [2.04,2.10]
        #onoff2 = [2.39,2.45]

        #this is H2 Cont magnitude
        #narrowmag1 = -2.5*np.log10(new_flux(obj[0],obj[1],0.0,0.0,onoff1,False)/new_flux(vega[0],vega[1],0.0,0.0,onoff1,False))
        #thus is IB2.42 magnitude
        #narrowmag2 = -2.5*np.log10(new_flux(obj[0],obj[1],0.0,0.0,onoff2,False)/new_flux(vega[0],vega[1],0.0,0.0,onoff2,False))

        if 'bs' in filetype or 'tl' in filetype:
            return broadmagJ,cusmagW,broadmagH,broadmagK,happ,chosenfile
        else:
            return broadmagJ,cusmagW,broadmagH,broadmagK
        
