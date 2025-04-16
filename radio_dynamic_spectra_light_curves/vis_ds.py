#!/usr/bin/env python
#
import numpy as np
import sys
import matplotlib.pyplot as plt
from astropy.time import Time
import matplotlib.cm as cm
from scipy.signal import medfilt
from scipy import optimize
from astropy.coordinates import SkyCoord, Angle, EarthLocation
from astropy import units as u

XX = 0
XY = 1
YX = 2
YY = 3

def std_iqr(x):
    """Robust estimation of the standard deviation, based on the inter-quartile
    (IQR) distance of x.
    This computes the IQR of x, and applies the Gaussian distribution
    correction, making it a consistent estimator of the standard-deviation
    (when the sample looks Gaussian with outliers).

    Parameters
    ----------
    x : `np.ndarray`
        Input vector

    Returns
    -------
    output : `float`
        A robust estimation of the standard deviation
    """
    from scipy.stats import iqr
    from scipy.special import erfinv
    good = x[np.where(np.isnan(x)==False)]
    correction = 2 ** 0.5 * erfinv(0.5)
    return correction * iqr(good) 

def time_str(t):
    t_val = Time(t/60.0/60.0/24.0, format='mjd', scale='utc')
    t_val.format='iso'
    return "%s" %(t_val)

# Perform RM-synthesis on Stokes Q and U data
#
# dataQ, dataU and freqs - contains the Q/U data at each frequency (in Hz) measured.
# startPhi, dPhi - the starting RM (rad/m^2) and the step size (rad/m^2)
def getFDF(dataQ, dataU, freqs, startPhi, stopPhi, dPhi, dType='float32'):
    # Calculate the RM sampling
    phiArr = np.arange(startPhi, stopPhi, dPhi)

    # Calculate the frequency and lambda sampling
    lamSqArr = np.power(2.99792458e8 / np.array(freqs), 2.0)

    # Calculate the dimensions of the output RM cube
    nPhi = len(phiArr)

    # Initialise the complex Faraday Dispersion Function (FDF)
    FDF = np.ndarray((nPhi), dtype='complex')

    # Assume uniform weighting
    wtArr = np.ones(len(lamSqArr), dtype=dType)

    K = 1.0 / np.nansum(wtArr)

    # Get the weighted mean of the LambdaSq distribution (B&dB Eqn. 32)
    lam0Sq = K * np.nansum(lamSqArr)

    # Mininize the number of inner-loop operations by calculating the
    # argument of the EXP term in B&dB Eqns. (25) and (36) for the FDF
    a = (-2.0 * 1.0j * phiArr)
    b = (lamSqArr - lam0Sq) 
    arg = np.exp( np.outer(a, b) )

    # Create a weighted complex polarised surface-brightness cube
    # i.e., observed polarised surface brightness, B&dB Eqns. (8) and (14)
    Pobs = (np.array(dataQ) + 1.0j * np.array(dataU))

    # Calculate the Faraday Dispersion Function
    # B&dB Eqns. (25) and (36)
    FDF = K * np.nansum(Pobs * arg, 1)
    return FDF, phiArr

def findpeaks(freqs, fdf, phi, rmsf, rmsfphi, nsigma):
    # Create the Gaussian filter for reconstruction
    c = 299792458.0 # Speed of light
    lam2 = (c / freqs) ** 2.0
    lam02 = np.mean(lam2)
    minl2 = np.min(lam2)
    maxl2 = np.max(lam2)
    width = (2.0 * np.sqrt(3.0)) / (maxl2 - minl2)

    Gauss = np.exp((-rmsfphi ** 2.0) / (2.0 * ((width / 2.355) ** 2.0)))
    components = np.zeros((len(phi)), np.float32)
    peaks = []
    phis = []
    std = 0.0
    rmsflen = int((len(rmsf) - 1) / 2)
    fdflen = len(phi) + rmsflen
    std = np.std(np.abs(fdf))
    peak1 = np.max(np.abs(fdf))
    pos1 = np.argmax(np.abs(fdf))
    val1 = phi[pos1]
    if peak1 > nsigma * std :
        fdf -= rmsf[rmsflen - pos1:fdflen - pos1] * fdf[pos1]
        peaks.append(peak1)
        phis.append(val1)
        components[pos1] += peak1
        std = np.std(np.abs(fdf))
    fdf += np.convolve(components, Gauss, mode='valid')
    return phis, peaks, std

def fitPL(fdata, sdata, serr):
#    # print('Running power-law fit:')
    good = np.where(np.isnan(sdata)==False)
    sdata = sdata[good]
    fdata = fdata[good]
    serr = serr[good]
    lognu = np.log10(fdata)
    logs = np.log10(sdata)

    logserr = serr
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit, args=(lognu, logs, logserr), full_output=1)
    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = np.sqrt( covar[1][1] )
    ampErr = np.sqrt( covar[0][0] ) * amp

    # print("   S_0=%f(+/-%f) alpha=%f(+/-%f)" %(amp, ampErr, index, indexErr))
    # print(amp,index)
    return amp, index #, (ampErr, indexErr))

def FourierShift(x, delta):
    # The size of the matrix.
    N = len(x)
    
    # FFT of our possibly padded input signal.
    X = np.fft.fft(x)
    
    # The mathsy bit. The floors take care of odd-length signals.
    x_arr = np.hstack([np.arange(np.floor(N/2), dtype=np.int), np.arange(np.floor(-N/2), 0, dtype=np.int)])

    x_shift = np.exp(-1j * 2 * np.pi * delta * x_arr / N)

#    x_shift = x_shift[None, :]   # Shape = (1, N)
    
    # Force conjugate symmetry. Otherwise this frequency component has no
    # corresponding negative frequency to cancel out its imaginary part.
    if np.mod(N, 2) == 0:
        x_shift[N//2] = np.real(x_shift[N//2])

    X = X * x_shift
    
    # Invert the FFT.
    xs = np.fft.ifft(X)
    
    # There should be no imaginary component (for real input
    # signals) but due to numerical effects some remnants remain.
    if np.isrealobj(x):
        xs = np.real(xs)
    
    return xs
    

class DynamicSpectra:
    def __init__(self, pickle_file, pbcorX=1.0, pbcorY=1.0, calCASA=True, ASKAPpolaxis=-45.0, swapASKAPXY=True, use_raw=False):
        obs_data = np.load("%s" %(pickle_file), allow_pickle=True, encoding='bytes')
        self.telescope = obs_data["TELESCOPE"]
        self.nchan = obs_data["NCHAN"]
        self.nint = obs_data["NINT"]
        self.nant = obs_data["NANT"]
        self.npol = obs_data["NPOL"]
        self.pols = obs_data["POLS"]
        self.ant_pos = obs_data["ANT_POS"]
        self.nant_avail = obs_data["NANT_AVAIL"]
        self.nbl = obs_data["NBL"]
        self.nbl_avail = obs_data["NBL_AVAIL"]
        self.freqs = obs_data["FREQS"] # frequency channels in Hz
        self.times = obs_data["TIMES"] # time stamps in seconds
        self.missing_ants = obs_data["MISSING_ANTS"]
        # Correct for primary beam at source location (if known)
        self.ds_std = obs_data["DS_STD"]
        self.ds_std[:,:,XX] /= (pbcorX*pbcorX)
        self.ds_std[:,:,XY] /= (pbcorX*pbcorY)
        self.ds_std[:,:,YX] /= (pbcorY*pbcorX)
        self.ds_std[:,:,YY] /= (pbcorY*pbcorY)
        self.ds_med = obs_data["DS_MED"]
        self.ds_med[:,:,XX] /= (pbcorX*pbcorX)
        self.ds_med[:,:,XY] /= (pbcorX*pbcorY)
        self.ds_med[:,:,YX] /= (pbcorY*pbcorX)
        self.ds_med[:,:,YY] /= (pbcorY*pbcorY)
        self.ds = obs_data["DS"]
        self.ds[:,:,XX] /= (pbcorX*pbcorX)
        self.ds[:,:,XY] /= (pbcorX*pbcorY)
        self.ds[:,:,YX] /= (pbcorY*pbcorX)
        self.ds[:,:,YY] /= (pbcorY*pbcorY)
        self.pbcorX = pbcorX
        self.pbcorY = pbcorY
        self.tintms = np.median(self.times[1:] - self.times[:-1])*1000.0 # integration time in mS
        self.calCASA = calCASA
        self.polaxis = ASKAPpolaxis
        self.swapASKAPXY = swapASKAPXY
        self.use_raw = use_raw
        self.direction = None
        

    def summary(self):
        print("Summary of observation:")
        print("Telescope: %s" %(self.telescope))
        print("%d channels (%.3f-%.3f MHz)" %(self.nchan, np.min(self.freqs)/1.0e6, np.max(self.freqs)/1.0e6))

        print("Antennas: %d" %(self.nant))
        print("Missing antennas: ", self.missing_ants)
        print("%d available antennas" %(self.nant_avail))
        print("Baselines: %d" %(self.nbl_avail))
        print("Visibilities: %d" %(self.nint * self.nbl_avail * self.nchan * self.npol))
        print("Integrations: %d (%s - %s)" %(self.nint, time_str(np.min(self.times)), time_str(np.max(self.times))))
        print("Integration time: %.1f mS" %(self.tintms))
        print("stt_date: %s" %(time_str(self.times[0]).split(' ')[0]))
        print("stt_time: %s" %(time_str(self.times[0]).split(' ')[1]))
        print("stt_imjd: %d" %(int(self.times[0]/86400.0)))
        print("stt_smjd: %d" %(int(86400*(self.times[0]/86400.0 - int(self.times[0]/86400.0)))))
        print("stt_offs: %d" %(86400.0*self.times[0]/86400.0 - int(86400.0*(self.times[0]/86400.0 - int(self.times[0]/86400.0 - int(self.times[0]/86400.0))))))
        print("Observation length: %.6f s" %(np.max(self.times) - np.min(self.times)))
        print("Channels: %d" %(self.nchan))
        print("Polarisations: %d\n" %(self.npol))

    def initiate_obj(self, coordstr=None):
        
        if isinstance(coordstr, str):
            if ":" in coordstr:
                racoord = Angle(coordstr.split(' ')[0], unit=u.hourangle)
                deccoord = Angle(coordstr.split(' ')[1], unit=u.deg)
                coord = SkyCoord(racoord, deccoord)
            elif 'h' in coordstr:
                coord = SkyCoord(coordstr)
            elif ' ' in coordstr:
                print('urgh')
        
        elif coordstr == None:
            coord = SkyCoord("04h37m15s -47d15m23s")
            
        else:
            coord = SkyCoord(*coordstr)
        
        
        self.direction = coord
        return coord
    
    
    
    def bary_time(self, direction):
        ASKAP_longitude = Angle("116:38:13.0", unit=u.deg)
        ASKAP_latitude = Angle("-26:41:46.0", unit=u.deg)
        ASKAP_location = EarthLocation(lat=ASKAP_latitude, lon=ASKAP_longitude)
        ASKAP_alt = 0.0

        MeerKAT_longitude = Angle("21:19:48.0", unit=u.deg)
        MeerKAT_latitude = Angle("-30:49:48.0", unit=u.deg)
        MeerKAT_location = EarthLocation(lat=MeerKAT_latitude, lon=MeerKAT_longitude)
        MeerKAT_alt = 0.0
        
        if self.telescope == "ASKAP":
            ut = Time(self.times/60.0/60.0/24.0, format='mjd', scale='utc', location=ASKAP_location)
        elif self.telescope == "MEERKAT":
            ut = Time(self.times/60.0/60.0/24.0, format='mjd', scale='utc', location=MeerKAT_location)

        ltt_bary = ut.light_travel_time(direction, ephemeris='jpl') # Convert to barycentric time
        batc = ut.tdb + ltt_bary
        bat_mjd = batc.value
        return bat_mjd

    
    def get_bary_time(self, coordstr = None):
        if self.direction == None:
            self.initiate_obj(coordstr)
        self.bat_mjd = self.bary_time(self.direction)
        return self.bat_mjd
    
    
    def mjd_time(self):
        ut = Time(self.times/60.0/60.0/24.0, format='mjd', scale='utc')
        return ut

    def dedisperse(self, DM = 0.0):
        # delay in mS; frequency in GHz
        dt_ms = 4.149 * DM *(1.0/np.power(self.freqs[-1]/1.0e9, 2.0) - 1.0/np.power(self.freqs/1.0e9, 2.0))
        dint = dt_ms / self.tintms # convert to samples
        # de-disperse
        for chan in range(self.nchan):
            for pol in [XX, XY, YX, YY]:
#                # print(self.freqs[chan], "GHz", self.pols[pol], dt[chan])
#                # print(self.ds[:,chan,pol])
                self.ds[:,chan,pol][np.where(np.isnan(self.ds[:,chan,pol]) == True)] = 0.0
                self.ds[:,chan,pol] = FourierShift(self.ds[:,chan,pol], dint[chan])
#                # print("%.3f MHz, %.1f sample shift" %(self.freqs[chan]/1.0e6, dint[chan]))
#                # print(self.ds[:,chan,pol])
#                offset = dt[self.nchan-chan - 1]
#                if offset == 0:
#                    continue
#                self.ds[:-offset,chan,pol] = self.ds[offset:,chan,pol]
    
    # Average over aT integrations and aF channels.
    # NOTE: this is a very rudimentary form of averaging and does not consider gaps between integrations.
    def average(self, aT = 1, aF = 1):
        # print("Original", self.ds.shape)
        nchan = self.nchan
        if (self.nchan % aF) != 0:
            # Spectrum doesn't divide up nicely, need to crop a bit
            naver = int(self.nchan / aF)
            nchan = int(naver * aF)
            self.ds = self.ds[:,:nchan,:]
            self.ds_std = self.ds_std[:,:nchan,:]
            self.ds_med = self.ds_med[:,:nchan,:]
            self.freqs = self.freqs[:nchan]
        nint = self.nint
        # print("Freq adjusted size: ",self.ds.shape)
        if (self.nint % aT) != 0:
            # Times don't divide up nicely, need to crop a bit
            naver = int(self.nint / aT)
            nint = int(naver * aT)
            self.ds = self.ds[:nint,:,:]
            self.ds_std = self.ds_std[:nint,:,:]
            self.ds_med = self.ds_med[:nint,:,:]
            self.times = self.times[:nint]
        # print("Time adjusted size: ", self.ds.shape)
        # Prepare axis labels for averaged data
        self.freqs = np.nanmean(self.freqs.reshape(int(self.nchan/aF), aF), axis=1)
        self.times = np.nanmean(self.times.reshape(int(self.nint/aT), aT), axis=1)
        self.nchan = nchan
        self.nint = nint
        # Average channels
        ds_aver_freq = np.nanmean(self.ds.reshape((self.nint, int(self.nchan/aF), aF, self.npol)), axis=2)
        # print("Freq averaged size: ",self.ds.shape)
        ds_aver_freq_std = np.nanmean(self.ds_std.reshape((self.nint, int(self.nchan/aF), aF, self.npol)), axis=2)
        ds_aver_freq_med = np.nanmean(self.ds_med.reshape((self.nint, int(self.nchan/aF), aF, self.npol)), axis=2)
        # Average times
        self.ds = np.nanmean(ds_aver_freq.reshape((int(self.nint/aT), aT, int(self.nchan/aF), self.npol)), axis=1)
        # print("Time averaged size: ",self.ds.shape)
        self.ds_std = np.nanmean(ds_aver_freq_std.reshape((int(self.nint/aT), aT, int(self.nchan/aF), self.npol)), axis=1)
        self.ds_med = np.nanmean(ds_aver_freq_med.reshape((int(self.nint/aT), aT, int(self.nchan/aF), self.npol)), axis=1)
        self.nchan = int(self.nchan/aF)
        self.nint = int(self.nint/aT)
        
    def get_stokes(self):
        if self.use_raw:
            It = np.real((self.ds[:,:,XX]+self.ds[:,:,YY]))
            Qt = np.real((self.ds[:,:,XX]-self.ds[:,:,YY]))
            Ut = np.real((self.ds[:,:,XY]+self.ds[:,:,YX]))
            Vt = np.imag((self.ds[:,:,XY]-self.ds[:,:,YX]))
            return It, Qt, Ut, Vt
            
        if (self.telescope == "ASKAP") and (self.calCASA == False): # ASKAP calibrated
            # print("Assuming ASKAP calibrated data")
            theta = 2.0 * np.radians(self.polaxis)
            if self.swapASKAPXY:
                # print("Swapping X and Y")
                It = np.real((self.ds[:,:,YY]+self.ds[:,:,XX]))
                Qt = np.real(np.cos(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) - np.sin(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))
                Ut = np.real(np.sin(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) + np.cos(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))
                Vt = np.imag((self.ds[:,:,YX]-self.ds[:,:,XY]))
            else:
                It = np.real((self.ds[:,:,XX]+self.ds[:,:,YY]))
                Qt = np.real(np.cos(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) - np.sin(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))
                Ut = np.real(np.sin(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) + np.cos(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))
                Vt = np.imag((self.ds[:,:,XY]-self.ds[:,:,YX]))
                
        elif (self.telescope == "ASKAP") and (self.calCASA == True): # CASA calibrated
            # print("Using CASA calibrated data")
            theta = 2.0 * np.radians(self.polaxis)
            if self.swapASKAPXY:
                # print("Swapping X and Y")
                It = np.real((self.ds[:,:,YY]+self.ds[:,:,XX]))/2.0
                Qt = np.real(np.cos(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) - np.sin(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))/2.0
                Ut = np.real(np.sin(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) + np.cos(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))/2.0
                Vt = np.imag((self.ds[:,:,YX]-self.ds[:,:,XY])/2.0)
            else:
                It = np.real((self.ds[:,:,XX]+self.ds[:,:,YY]))/2.0
                Qt = np.real(np.cos(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) - np.sin(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))/2.0
                Ut = np.real(np.sin(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) + np.cos(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))/2.0
                Vt = np.imag((self.ds[:,:,XY]-self.ds[:,:,YX])/2.0)
        elif self.telescope == "MWA": # CASA calibrated
            It = np.real((self.ds[:,:,XX]+self.ds[:,:,YY])/2.0)
            Qt = np.real((self.ds[:,:,XX]-self.ds[:,:,YY])/2.0)
            Ut = np.real((self.ds[:,:,XY]+self.ds[:,:,YX])/2.0)
            Vt = np.imag((self.ds[:,:,XY]-self.ds[:,:,YX])/2.0)
        elif self.telescope == "MeerKAT": # CASA calibrated
            It = np.real((self.ds[:,:,XX]+self.ds[:,:,YY])/2.0)
            Qt = np.real((self.ds[:,:,XX]-self.ds[:,:,YY])/2.0)
            Ut = np.real((self.ds[:,:,XY]+self.ds[:,:,YX])/2.0)
            Vt = np.imag((self.ds[:,:,XY]-self.ds[:,:,YX])/2.0)
        elif self.telescope == "GMRT": # CASA calibrated
            It = np.real((self.ds[:,:,XX]+self.ds[:,:,YY])/2.0)
            Qt = np.real((self.ds[:,:,XX]-self.ds[:,:,XX])/2.0)
            Ut = np.real((self.ds[:,:,XY]+self.ds[:,:,YY])/2.0)
            Vt = np.imag((self.ds[:,:,XX]-self.ds[:,:,YY])/2.0)
        return It, Qt, Ut, Vt
    
    
    def get_stokes_imag(self):
        if self.use_raw:
            It = np.imag((self.ds[:,:,XX]+self.ds[:,:,YY]))
            Qt = np.imag((self.ds[:,:,XX]-self.ds[:,:,YY]))
            Ut = np.imag((self.ds[:,:,XY]+self.ds[:,:,YX]))
            Vt = np.real((self.ds[:,:,XY]-self.ds[:,:,YX]))
            return It, Qt, Ut, Vt
            
        if (self.telescope == "ASKAP") and (self.calCASA == False): # ASKAP calibrated
            # print("Assuming ASKAP calibrated data")
            theta = 2.0 * np.radians(self.polaxis)
            if self.swapASKAPXY:
                # print("Swapping X and Y")
                It = np.imag((self.ds[:,:,YY]+self.ds[:,:,XX]))
                Qt = np.imag(np.cos(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) - np.sin(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))
                Ut = np.imag(np.sin(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) + np.cos(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))
                Vt = np.real((self.ds[:,:,YX]-self.ds[:,:,XY]))
            else:
                It = np.imag((self.ds[:,:,XX]+self.ds[:,:,YY]))
                Qt = np.imag(np.cos(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) - np.sin(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))
                Ut = np.imag(np.sin(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) + np.cos(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))/2.0
                Vt = np.real((self.ds[:,:,XY]-self.ds[:,:,YX]))
                
        elif (self.telescope == "ASKAP") and (self.calCASA == True): # CASA calibrated
            # print("Using CASA calibrated data")
            theta = 2.0 * np.radians(self.polaxis)
            if self.swapASKAPXY:
                # print("Swapping X and Y")
                It = np.imag((self.ds[:,:,YY]+self.ds[:,:,XX]))/2.0
                Qt = np.imag(np.cos(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) - np.sin(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))/2.0
                Ut = np.imag(np.sin(theta)*(self.ds[:,:,YX]+self.ds[:,:,XY]) + np.cos(theta)*(self.ds[:,:,YY]-self.ds[:,:,XX]))/2.0
                Vt = np.real((self.ds[:,:,YX]-self.ds[:,:,XY])/2.0)
            else:
                It = np.imag((self.ds[:,:,XX]+self.ds[:,:,YY]))/2.0
                Qt = np.imag(np.cos(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) - np.sin(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))/2.0
                Ut = np.imag(np.sin(theta)*(self.ds[:,:,XY]+self.ds[:,:,YX]) + np.cos(theta)*(self.ds[:,:,XX]-self.ds[:,:,YY]))/2.0
                Vt = np.real((self.ds[:,:,XY]-self.ds[:,:,YX])/2.0)
        elif self.telescope == "MWA": # CASA calibrated
            It = np.imag((self.ds[:,:,XX]+self.ds[:,:,YY])/2.0)
            Qt = np.imag((self.ds[:,:,XX]-self.ds[:,:,YY])/2.0)
            Ut = np.imag((self.ds[:,:,XY]+self.ds[:,:,YX])/2.0)
            Vt = np.real((self.ds[:,:,XY]-self.ds[:,:,YX])/2.0)
        elif self.telescope == "MeerKAT": # CASA calibrated
            It = np.imag((self.ds[:,:,XX]+self.ds[:,:,YY])/2.0)
            Qt = np.imag((self.ds[:,:,XX]-self.ds[:,:,YY])/2.0)
            Ut = np.imag((self.ds[:,:,XY]+self.ds[:,:,YX])/2.0)
            Vt = np.real((self.ds[:,:,XY]-self.ds[:,:,YX])/2.0)
        elif self.telescope == "GMRT": # CASA calibrated
            It = np.imag((self.ds[:,:,XX]+self.ds[:,:,YY])/2.0)
            Qt = np.imag((self.ds[:,:,XX]-self.ds[:,:,XX])/2.0)
            Ut = np.imag((self.ds[:,:,XY]+self.ds[:,:,YY])/2.0)
            Vt = np.real((self.ds[:,:,XX]-self.ds[:,:,YY])/2.0)
        return It, Qt, Ut, Vt
    
    def set_rm(self, RM):
        self.RM = RM
        # print(self.RM)
        return RM
    
    
    def defaraday(self, RM = None):
        C = 299792458   # Speed of light

        I, Q, U, V = self.get_stokes() #dynamic spectra
        
        L = Q + 1j * U
        lambdas = C / self.freqs
        L_df = np.exp(1j * 2.0 * ( - self.RM * lambdas **2.0)[None, :])*L
        return L_df
    
    
#     def get_linpol_PA(self, RM):
        
#         C = 299792458   # Speed of light

#         I, Q, U, V = self.get_stokes() #dynamic spectra
#         lambdas = C / self.freqs

#         PA = 0.5 * np.arctan2(np.real(U), np.real(Q))

#         PA_df = np.arctan2(np.tan((PA - (RM*lambdas**2.0)[None, :])), np.ones_like(PA))
        
#         L_df = self.defaraday(RM = RM)
#         PA_lc = 0.5 * np.arctan2(np.nanmean(L_df.imag, axis = 1), np.nanmean(L_df.real, axis = 1))
    
#         return PA_df, PA_lc
        
    
    
    def flagNoisyVChannels(self, nsigma):
        It, Qt, Ut, Vt = self.get_stokes()
        vstd = np.nanstd(Vt, axis=0)
        bad_chans = np.where(vstd > (nsigma * np.nanmedian(vstd)))
        for chan in bad_chans[0]:
            # print("flag channel %d" %(chan))
            self.ds[:,chan,XX] = np.nan
            self.ds[:,chan,XY] = np.nan
            self.ds[:,chan,YX] = np.nan
            self.ds[:,chan,YY] = np.nan

    # Flag based on Stokes V extremes (usually a good indicator of RFI in the absense of true circular polarisation)
    def flagV(self, nsigma = 3.0):
        It, Qt, Ut, Vt = self.get_stokes()
        vstd = np.nanstd(Vt)
        bad = np.where(np.abs(Vt) > nsigma*vstd)
        self.ds[:,:,XX][bad] = np.nan
        self.ds[:,:,XY][bad] = np.nan
        self.ds[:,:,YX][bad] = np.nan
        self.ds[:,:,YY][bad] = np.nan
        
        for chan in range(self.nchan):
            frac = float(np.sum(np.isnan(self.ds[:,chan,XX])))/float(self.nchan)
            if frac > 0.05:
                print("ods.flag_chan(%d,%d)" %(chan, chan))

    # Flag based on Stokes V extremes (usually a good indicator of RFI in the absense of true circular polarisation)
    def flagQU(self, nsigma = 3.0):
        It, Qt, Ut, Vt = self.get_stokes()
        qstd = np.nanstd(Qt)
        bad = np.where(np.abs(Qt) > nsigma*qstd)
        self.ds[:,:,XX][bad] = np.nan
        self.ds[:,:,XY][bad] = np.nan
        self.ds[:,:,YX][bad] = np.nan
        self.ds[:,:,YY][bad] = np.nan
        ustd = np.nanstd(Ut)
        bad = np.where(np.abs(Ut) > nsigma*ustd)
        self.ds[:,:,XX][bad] = np.nan
        self.ds[:,:,XY][bad] = np.nan
        self.ds[:,:,YX][bad] = np.nan
        self.ds[:,:,YY][bad] = np.nan
        
    # Flag based on Stokes V extremes (usually a good indicator of RFI in the absense of true circular polarisation)
    def flagI(self, nsigma = 3.0):
        It, Qt, Ut, Vt = self.get_stokes()
        istd = np.nanstd(It)
        bad = np.where(np.abs(It) > nsigma*istd)
        self.ds[:,:,XX][bad] = np.nan
        self.ds[:,:,XY][bad] = np.nan
        self.ds[:,:,YX][bad] = np.nan
        self.ds[:,:,YY][bad] = np.nan
        
    # Flag channel range from channel c1 to c2
    def flag_chan(self, c1, c2):
        self.ds[:,c1:c2,:] = np.nan
        
    # Flag channel range from channel c1 to c2
    def flag_time(self, t1, t2):
        self.ds[t1:t2,:,:] = np.nan

    # Flag channel range from channel c1 to c2
    def flag_window(self, t1, t2, c1, c2):
        self.ds[t1:t2,c1:c2,:] = np.nan

    def get_lc(self):
        I, Q, U, V = self.get_stokes()
        averI = np.nanmean(I, axis=1)  # Average DS in frequency
        return averI
                                                                                                                      
    def get_stokes_lc(self):
        I, Q, U, V = self.get_stokes()
        averI = np.nanmean(I, axis=1)  # Average DS in frequency
        averQ = np.nanmean(Q, axis=1)  # Average DS in frequency
        averU = np.nanmean(U, axis=1)  # Average DS in frequency
        averV = np.nanmean(V, axis=1)  # Average DS in frequency

        return averI, averQ, averU, averV
    
    def get_lc_imag(self):
        I, Q, U, V = self.get_stokes_imag()
        averI = np.nanmean(I, axis=1)  # Average DS in frequency
        return averI

    def get_stokes_lc_imag(self):
        I, Q, U, V = self.get_stokes_imag()
        averI = np.nanmean(I, axis=1)  # Average DS in frequency
        averQ = np.nanmean(Q, axis=1)  # Average DS in frequency
        averU = np.nanmean(U, axis=1)  # Average DS in frequency
        averV = np.nanmean(V, axis=1)  # Average DS in frequency

        return averI, averQ, averU, averV
                                                                                                                      

    def get_rms_ds(self):
        I, Q, U, V = self.get_stokes_imag()
        rmsI = np.nanstd(I)
        rmsQ = np.nanstd(Q)
        rmsU = np.nanstd(U)
        rmsV = np.nanstd(V)

        return rmsI, rmsQ, rmsU, rmsV 
                                                 
                                                                                                                      
    def get_rms_lc(self):
        I, Q, U, V = self.get_stokes_lc_imag()
        rmsI = np.nanstd(I)
        rmsQ = np.nanstd(Q)
        rmsU = np.nanstd(U)
        rmsV = np.nanstd(V)

        return rmsI, rmsQ, rmsU, rmsV                                                                                                                      
 
                                                                                                                      
    # Plot the time-series light curve averaged over the band
    def plot_rawsed(self, t):
        xx = self.ds[t,:,XX]
        yy = self.ds[t,:,YY]
        stdI = np.nanstd(xx)
        # print(np.nanmean(np.real(xx+yy)))
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot(111)
        plot, = ax1.plot(self.freqs/1.0e9, np.real(xx), marker='', color="black", label="XX")
        plot, = ax1.plot(self.freqs/1.0e9, np.real(yy), marker='', color="red", label="YY")
        plot, = ax1.plot(self.freqs/1.0e9, np.real(xx+yy), marker='', color="green", label="XX+YY")
        plot, = ax1.plot(self.freqs/1.0e9, np.real(xx-yy), marker='', color="blue", label="XX-YY")
        ax1.set_title("Raw light Curve")
        ax1.set_xlabel("Integration")
        ax1.set_ylabel("Flux Density (mJy)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    # Plot the time-series light curve averaged over the band
    def plot_lc(self, nsigma, sub_med=True, real_time=True, ppol = "IQUVP"):
        I, Q, U, V = self.get_stokes()
        P = np.sqrt(Q*Q + U*U)
        averI = np.nanmean(I, axis=1) * 1000.0  # Average DS in frequency and convert to mJy
        averQ = np.nanmean(Q, axis=1) * 1000.0
        averU = np.nanmean(U, axis=1) * 1000.0
        averV = np.nanmean(V, axis=1) * 1000.0

        averP = np.nanmean(P, axis=1) * 1000.0

        stdI = np.nanstd(averI)
        if real_time == True:
            t = self.times - self.times[0]
        else:
            t = range(len(self.times))

        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot(111)
        if ppol.find("I") != -1:
            medI = medfilt(averI, 11)
            if sub_med == True:
                plot, = ax1.plot(t, averI - medI, marker='', color="black", label="I")
            else:
                plot, = ax1.plot(t, averI, marker='', color="black", label="I")
        if ppol.find("Q") != -1:
            medQ = medfilt(averQ, 11)
            if sub_med == True:
                plot, = ax1.plot(t, averQ - medQ, marker='', color="red", label="Q")
            else:
                plot, = ax1.plot(t, averQ, marker='', color="red", label="Q")
        if ppol.find("U") != -1:
            medU = medfilt(averU, 11)
            if sub_med == True:
                plot, = ax1.plot(t, averU - medU, marker='', color="blue", label="U")
            else:
                plot, = ax1.plot(t, averU, marker='', color="blue", label="U")
        if ppol.find("V") != -1:
            medV = medfilt(averV, 11)
            if sub_med == True:
                plot, = ax1.plot(t, averV - medV, marker='', color="green", label="V")
            else:
                plot, = ax1.plot(t, averV, marker='', color="green", label="V")
        if ppol.find("P") != -1:
            medP = medfilt(averP, 11)
            if sub_med == True:
                plot, = ax1.plot(t, averP - medP, marker='', color="gray", label="P")
            else:
                plot, = ax1.plot(t, averP, marker='', color="gray", label="P")
        ax1.set_title("Light Curve")
        if real_time == True:
            ax1.set_xlabel("Elapsed time (s)")
        else:
            ax1.set_xlabel("Integration")
        ax1.set_ylabel("Flux Density (mJy)")
        ax1.set_ylim(-nsigma*stdI/6.0, nsigma*stdI)
        # print("stdI=%.3f" %(stdI))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_ILVPA(self):
        
        #makes a plot in the style of pav -SFT
        
        
        
        x = self.times - self.times[0] #topocentric time from obs start in seconds
        PA_lc, PA_e = self.get_linpol_pa_lc()

        L_df = self.defaraday()
        
        L_lc = np.abs(np.nanmean(L_df, axis=1))
        
        
        I_lc, Q_lc, U_lc, V_lc = self.get_stokes_lc()
        rmsI, rmsQ, rmsU, rmsV = self.get_rms_lc()
        
        PA_lc[np.abs(PA_e) > 20.0] = np.nan
        PA_lc[L_lc < 4.0*rmsU/np.sqrt(2)] = np.nan
 
        fig, (ax1, ax2) = plt.subplots(2,1,gridspec_kw = {"height_ratios": [1, 3]}, figsize=(6,8), sharex = True)
        
        plt.sca(ax1)
        plt.errorbar(x, PA_lc, yerr = PA_e, c =  'k', fmt='o', markersize=4, ecolor='0.5')
        plt.xlabel("Time (s)")
        plt.ylabel("P. A. (deg)")
        plt.axhline(-90.0, ls = '--', c = 'k', alpha = 0.3, lw = 1.0)
        plt.axhline(90.0, ls = '--', c = 'k' , alpha = 0.3, lw = 1.0)
        plt.ylim(-98, 98)

        # plt.xlim(x[950], x[1250])

        #in case baseline subtraction is not done properly, subtract the low-flux mean off the lightcurve
        I_lc_mean = np.nanmean(I_lc[(I_lc) < 0.005])
        
        # ax2 = fig.add_subplot(212, sharex = ax1)
        plt.sca(ax2)
        plt.plot(x, I_lc-I_lc_mean, label = "I", c= 'k', zorder=0)
        plt.plot(x, L_lc, label = "L", c = 'r')
        plt.plot(x, V_lc, label = "V", c = 'b')

        plt.xlabel("Time (s)")
        plt.ylabel("Flux density (Jy)")
        plt.legend()
        fig.subplots_adjust(hspace=0)

        #plt.show()
        return fig
        
        
    # Plot the time-series light curve averaged over the band
    def lc_peaks(self, nsigma, sub_med=False):
        # print(nsigma)
        I, Q, U, V = self.get_stokes()
        P = np.sqrt(Q*Q + U*U)
        averI = np.nanmean(I, axis=1) * 1000.0  # Average DS in frequency and convert to mJy
        averQ = np.nanmean(Q, axis=1) * 1000.0
        averU = np.nanmean(U, axis=1) * 1000.0
        averV = np.nanmean(V, axis=1) * 1000.0
        averP = np.nanmean(P, axis=1) * 1000.0
        if sub_med == True:
            medI = medfilt(averI, 11)
            averI -= medI

        stdI = std_iqr(averI)
        # print(stdI, nsigma)
        t = []
        for index in range(len(self.times)):
            if averI[index] > nsigma * stdI:
                t.append(index)
                # print("%s,%d,%.3f" %(time_str(self.times[index]), index, averI[index]))
        return t
    
    # Plot the dynamic spectra for the four polarisations
    def plot_ds(self, sigma, real_time=False, real_freq=False):
        It, Qt, Ut, Vt = self.get_stokes()
        vstd = np.nanstd(Vt)

        current_cmap = cm.get_cmap("cubehelix")#.copy()
        current_cmap.set_bad(color=current_cmap(0.5))
        
        if real_time==False and real_freq==False:
            ext = [0, self.nint, 0, self.nchan]
            tlabel = "Integration"
            flabel = "Chan"
        elif real_time==True and real_freq==False:
            ext = [0.0, self.times[-1]-self.times[0], 0, self.nchan]
            tlabel = "Elapsed Time (s)"
            flabel = "Chan"
        elif real_time==False and real_freq==True:
            tlabel = "Integration"
            flabel = "$\\nu$ (GHz)"
            ext = [0, self.nint, self.freqs[0],self.freqs[-1]]
        else:
            ext = [0.0, self.times[-1]-self.times[0], self.freqs[0],self.freqs[-1]]
            tlabel = "Elapsed Time (s)"
            flabel = "$\\nu$ (GHz)"

        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(221)
        ax1.set_title("Dynamic Spectra (I)")
        ax1.set_xlabel(tlabel)
        ax1.set_ylabel(flabel)
        plt.imshow(np.transpose(It), origin='lower', clim=(0.0, sigma*vstd), extent=ext, aspect="auto", cmap=current_cmap)
        cbar = plt.colorbar()
        cbar.set_label('Jy')#,labelpad=-75)

        ax2 = fig.add_subplot(222)
        ax2.set_title("Dynamic Spectra (Q)")
        ax2.set_xlabel(tlabel)
        ax2.set_ylabel(flabel)
        plt.imshow(np.transpose(Qt), origin='lower', clim=(-sigma*vstd, sigma*vstd), extent=ext, aspect="auto", cmap=current_cmap)
        cbar = plt.colorbar()
        cbar.set_label('Jy')#,labelpad=-75)

        ax3 = fig.add_subplot(223)
        ax3.set_title("Dynamic Spectra (U)")
        ax3.set_xlabel(tlabel)
        ax3.set_ylabel(flabel)
        plt.imshow(np.transpose(Ut), origin='lower', clim=(-sigma*vstd, sigma*vstd), extent=ext, aspect="auto", cmap=current_cmap)
        cbar = plt.colorbar()
        cbar.set_label('Jy')#,labelpad=-75)

        ax4 = fig.add_subplot(224)
        ax4.set_title("Dynamic Spectra (V)")
        ax4.set_xlabel(tlabel)
        ax4.set_ylabel(flabel)
        plt.imshow(np.transpose(Vt), origin='lower', clim=(-sigma*vstd, sigma*vstd), extent=ext, aspect="auto", cmap=current_cmap)
        cbar = plt.colorbar()
        cbar.set_label('Jy')#,labelpad=-75)

        plt.tight_layout()
        plt.show()
        plt.close()
        
    # Plot the dynamic spectra for the four polarisations
    def plot_cpds(self, sigma, sname = "", real_time=False, real_freq=False):
        It, Qt, Ut, Vt = self.get_stokes()
        vstd = np.nanstd(Vt)

        current_cmap = plt.get_cmap("cubehelix")#.copy()
        current_cmap.set_bad(color=current_cmap(0.5))
        
        if real_time==False and real_freq==False:
            ext = [0, self.nint, 0, self.nchan]
            tlabel = "Integration"
            flabel = "Chan"
        elif real_time==True and real_freq==False:
            ext = [0.0, self.times[-1]-self.times[0], 0, self.nchan]
            tlabel = "Elapsed Time (s)"
            flabel = "Chan"
        elif real_time==False and real_freq==True:
            tlabel = "Integration"
            flabel = "$\\nu$ (GHz)"
            ext = [0, self.nint, self.freqs[0],self.freqs[-1]]
        else:
            ext = [0.0, self.times[-1]-self.times[0], self.freqs[0],self.freqs[-1]]
            tlabel = "Elapsed Time (s)"
            flabel = "$\\nu$ (GHz)"

        fig = plt.figure(figsize=(14, 8))
        ax4 = fig.add_subplot(111)
        ax4.set_title("Dynamic Spectra (V) - %s" %(sname))
        ax4.set_xlabel(tlabel)
        ax4.set_ylabel(flabel)
        plt.imshow(np.transpose(Vt), origin='lower', clim=(-sigma*vstd, sigma*vstd), extent=ext, aspect="auto", cmap=current_cmap)
        cbar = plt.colorbar()
        cbar.set_label('Jy')#,labelpad=-75)

        plt.tight_layout()
        if len(sname):
            plt.savefig("%s.png" %(sname))
        else:
            plt.show()
        plt.close()
        
    # Plot the SED for the given time integration
    def plot_sed(self, t, pols="IQUV", title="SED", plotwl2=False):
        I, Q, U, V = self.get_stokes()
        It = I[t]
        Qt = Q[t]
        Ut = U[t]
        Vt = V[t]
        Pt = np.sqrt(Qt*Qt+Ut*Ut)

        if plotwl2:
            xaxis = np.power(3.0e8/self.freqs, 2.0)
        else:
            xaxis = self.freqs/1.0e9
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot(111)
        if "I" in pols:
            plot, = ax1.plot(xaxis, It*1000.0, marker='', color="black", label="I")
        if "Q" in pols:
            plot, = ax1.plot(xaxis, Qt*1000.0, marker='', color="red", label="Q")
        if "XX" in pols:
            plot, = ax1.plot(xaxis, self.ds[0,:,XX]*1000.0, marker='', color="red", label="XX", ls=":")
        if "YY" in pols:
            plot, = ax1.plot(xaxis, self.ds[0,:,YY]*1000.0, marker='', color="blue", label="YY", ls=":")
        if "U" in pols:
            plot, = ax1.plot(xaxis, Ut*1000.0, marker='', color="blue", label="U")
        if "V" in pols:
            plot, = ax1.plot(xaxis, Vt*1000.0, marker='', color="green", label="V")
        if "P" in pols:
            plot, = ax1.plot(xaxis, Pt*1000.0, marker='', color="gray", label="P")
        ax1.set_title(title)
        if plotwl2:
            ax1.set_xlabel("(m$^{2}$)")
        else:
            ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("Flux Density (mJy)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    def get_rm_ppa(self, t, min_snr=10., startPhi=-120., dPhi=0.05, nsigma=3.):
        stopPhi = -startPhi+dPhi
        
        I, Q, U, V = self.get_stokes()
        It = I[t]; Qt = Q[t]; Ut = U[t]; Vt = V[t]
        I_mean = np.nanmean(It)

        FDFqu, phi = getFDF(Qt, Ut, self.freqs, startPhi, stopPhi, dPhi)
        rstartPhi = startPhi * 2
        rstopPhi = stopPhi * 2 - dPhi
        RMSF, rmsfphi = getFDF(np.ones(Qt.shape), np.zeros(Ut.shape), self.freqs, rstartPhi, rstopPhi, dPhi)

        phis, peaks, sigma = findpeaks(self.freqs, FDFqu, phi, RMSF, rmsfphi, nsigma)
        return phis[0], peaks[0], sigma
    
    def find_fdf_peaks(self, min_snr = 10.0, startPhi = -1000.0, dPhi = 1.0):
        t_vals = []
        pi_vals = []
        phi_vals = []
        snr_vals = []
        I, Q, U, V = self.get_stokes()
        stopPhi = -startPhi+dPhi

        chanBW = np.median(self.freqs[1:] - self.freqs[:-1])
        C = 299792458   # Speed of light

        fmin = np.min(self.freqs)
        fmax = np.max(self.freqs)
        bw = fmax - fmin

        dlambda2 = np.power(C / fmin, 2) - np.power(C / (fmin + chanBW), 2)
        Dlambda2 = np.power(C / fmin, 2) - np.power(C / (fmin + bw), 2)
        phimax = np.sqrt(3) / dlambda2
        dphi = 2.0 * np.sqrt(3) / Dlambda2
        phiR = dphi / 5.0
        Nphi = 2 * phimax / phiR
        fwhm = dphi
    
        for t in range(self.nint):
            It = I[t]
            Qt = Q[t]
            Ut = U[t]
            Vt = V[t]
            I_mean = np.nanmean(It)

            # dirty, phi = getFDF(Qt, Ut, self.freqs, startPhi, stopPhi, dPhi)
            FDFqu, phi = getFDF(Qt, Ut, self.freqs, startPhi, stopPhi, dPhi)
            rstartPhi = startPhi * 2
            rstopPhi = stopPhi * 2 - dPhi
            RMSF, rmsfphi = getFDF(np.ones(Qt.shape), np.zeros(Ut.shape), self.freqs, rstartPhi, rstopPhi, dPhi)

            # Do a very rudimentary clean i.e. find a peak and subtract out the RMSF at that peak
            phis, peaks, sigma = findpeaks(self.freqs, FDFqu, phi, RMSF, rmsfphi, 6.0)
            if len(peaks) > 0:
                snr = peaks[0] / sigma
                if snr > min_snr:
                    t_vals.append(t)
                    pi_vals.append(peaks[0])
                    snr_vals.append(snr)
                    phi_vals.append(phis[0])
#                    # print("%s PI=%.3f mJy/beam (SNR %.1f) RM=%.1f" %(time_str(ods.times[t]), 1000.0*peaks[0], snr, phis[0]))
        return np.array(t_vals), np.array(pi_vals), np.array(snr_vals), np.array(phi_vals)

    def get_sed(self, t):
        I, Q, U, V = self.get_stokes()
        It = I[t]
        Qt = Q[t]
        Ut = U[t]
        Vt = V[t]
        return It, Qt, Ut, Vt
        
    def get_sed_aver(self):
        I, Q, U, V = self.get_stokes()
        It = np.nanmean(I, axis=0) 
        Qt = np.nanmean(Q, axis=0) 
        Ut = np.nanmean(U, axis=0) 
        Vt = np.nanmean(V, axis=0) 
        return It, Qt, Ut, Vt
        
    def get_fdf(self, t, startPhi = -1000.0, dPhi = 1.0, nsigma=3.):
        I, Q, U, V = self.get_stokes()
        It = I[t]
        Qt = Q[t]
        Ut = U[t]
        Vt = V[t]
        I_mean = np.nanmean(It)
        stopPhi = -startPhi+dPhi

        chanBW = np.median(self.freqs[1:] - self.freqs[:-1])
        C = 299792458   # Speed of light

        fmin = np.min(self.freqs)
        fmax = np.max(self.freqs)
        bw = fmax - fmin


        dlambda2 = np.power(C / fmin, 2) - np.power(C / (fmin + chanBW), 2)
        Dlambda2 = np.power(C / fmin, 2) - np.power(C / (fmin + bw), 2)
        phimax = np.sqrt(3) / dlambda2
        dphi = 2.0 * np.sqrt(3) / Dlambda2
        phiR = dphi / 5.0
        Nphi = 2 * phimax / phiR
        fwhm = dphi
        
        dirty, phi = getFDF(Qt, Ut, self.freqs, startPhi, stopPhi, dPhi)
        FDFqu, phi = getFDF(Qt, Ut, self.freqs, startPhi, stopPhi, dPhi)
        rstartPhi = startPhi * 2
        rstopPhi = stopPhi * 2 - dPhi
        RMSF, rmsfphi = getFDF(np.ones(Qt.shape), np.zeros(Ut.shape), self.freqs, rstartPhi, rstopPhi, dPhi)

        # Do a very rudimentary clean i.e. find a peak and subtract out the RMSF at that peak
        phis, peaks, sigma = findpeaks(self.freqs, FDFqu, phi, RMSF, rmsfphi, nsigma)
        return FDFqu, I_mean, phi, phis, peaks, sigma, fwhm

    # Plot the Faraday Dispersion Function for the given time integration
    def plot_fdf(self, t, startPhi = -1000.0, dPhi = 1.0, doplot=False):
        FDFqu, I_mean, phi, phis, peaks, sigma, fwhm = self.get_fdf(t, startPhi, dPhi)
        snr = 0.0
        if len(peaks) > 0:#
            snr = peaks[0] / sigma
            phierr = fwhm / (2 * snr)
            # print("%s PI: %.3f mJy/beam Imean=%.3f (SNR=%.1f) phi=%.3f+/-%.1f fp=%.1f%%" %(time_str(self.times[t]), peaks[0]*1000.0, I_mean*1000.0, snr, phis[0], phierr, 100.0*peaks[0] / I_mean))
        if doplot:
            # Plot the RMSF
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            plot, = ax1.plot(phi, 1000.0*np.abs(FDFqu), marker=None, color="black")
            ax1.set_title("FDF - cleaned")
            ax1.set_xlabel("phi (rad m$^{-2}$)")
            ax1.set_ylabel("Flux (mJy beam$^{-1}$ RMSF$^{-1}$)")
            ax1.set_xlim(phi[0], phi[-1])
            plt.show()
            plt.close()
        return snr

    def get_linpol_pa_ds(self, defaraday=True):
        
        L_df = self.defaraday()
        
        PA_ds = 0.5 * np.arctan2(L_df.imag, L_df.real)*180/np.pi
        
        I, Q, U, V = self.get_stokes()
        I_rms, Q_rms, U_rms, V_rms = self.get_rms_ds()
        
        Q, U = (L_df.real, L_df.imag)
        
        P = np.abs(L_df)/I
        
        sig_P = np.sqrt((1.0 / (P*I**2.0)**2.0)*(Q**2 * Q_rms**2.0 + U**2.0 * U_rms**2.0 + (P**2 * I)**2*I_rms**2.0))
        
        PA_e = 180.0/np.pi * np.sqrt((Q**2 * U_rms**2 + U**2 * Q_rms**2)/(Q**2*Q_rms**2 + U**2 * U_rms**2.0))*0.5*sig_P/P

        return PA_ds, PA_e
    
    def get_linpol_pa_lc(self, defaraday=True):
        I, Q, U, V = self.get_stokes_lc()
        I_rms, Q_rms, U_rms, V_rms = self.get_rms_lc()
        
        L_df = self.defaraday()
        
        L_df_lc = np.nanmean(L_df, axis = 1)
        
        Q, U = (L_df_lc.real, L_df_lc.imag)
        
        PA_lc = 0.5 * np.arctan2(U, Q)*180/np.pi
        
        
        
        P = np.abs(L_df_lc)/I
        
        sig_P = np.sqrt((1.0 / (P*I**2.0)**2.0)*(Q**2 * Q_rms**2.0 + U**2.0 * U_rms**2.0 + (P**2 * I)**2*I_rms**2.0))
        
        PA_e = 180.0/np.pi * np.sqrt((Q**2 * U_rms**2 + U**2 * Q_rms**2)/(Q**2*Q_rms**2 + U**2 * U_rms**2.0))*0.5*sig_P/P

        return PA_lc, PA_e
    
    def get_spidx_lc(self,):
        spidx = []; spidx_err = []
        for t in range(self.nint):
            s, serr = self.spidx(t)
            spidx.append(s)
            spidx_err.append(serr)
        return np.array(spidx), np.array(spidx_err)
    
    def calc_phase(self, F0, F1, PEPOCH):
        
        P0 = 1/F0
        
        bat_mjd = self.get_bary_time()
        bat_mjd_s = bat_mjd*86400.0
        x = bat_mjd_s - PEPOCH*86400.0
        
        phase = F0 / 1 * x**1 + F1 / 2 * x**2.0
        phase -= phase[0]
        phase += 0.5
        self.phase = phase%1
        return phase%1, phase
    
    
    def fold(self, F0, F1, PEPOCH, nbin, RM):

        phase, realphase = self.calc_phase(F0, F1, PEPOCH)
        nturn = np.floor(realphase).astype(int)
        maxNturn = max(nturn) +1
        # print(self.phase)
        I, Q, U, V = self.get_stokes()
        
        L_df = self.defaraday(RM=self.RM)
        PA_df, PA_e = self.get_linpol_pa_ds()
        L_abs = np.abs(L_df)
        # print('here')
        I_flux_phase = np.zeros((nbin, maxNturn, self.nchan))
        L_flux_phase = np.zeros((nbin, maxNturn, self.nchan))
        Q_flux_phase = np.zeros((nbin, maxNturn, self.nchan))
        U_flux_phase = np.zeros((nbin, maxNturn, self.nchan))
        V_flux_phase = np.zeros((nbin, maxNturn, self.nchan))
        PA_flux_phase = np.zeros((nbin, maxNturn, self.nchan))
        
        # print('here')

        counts = np.zeros((nbin, maxNturn, self.nchan))
        # print("counts done")
                                 
        # print(len(self.phase))
        # print(type(self.phase))
        
        phasebins = np.linspace(0, 1, nbin)

        for ind, p_ in enumerate(self.phase):
#             # print("{:.3f}".format(ind/len(self.phase)))
            phasebin_ind = np.argmin(np.abs(p_ - phasebins))
            phasebin = phasebins[phasebin_ind]
            nturn_ = int(realphase[ind])
            phasebin_ind = np.argmin(np.abs(p_ - phasebins))
            I_flux_phase[phasebin_ind, nturn_, :] += I[ind, :] 
            L_flux_phase[phasebin_ind, nturn_, :] += L_abs[ind, :]
            Q_flux_phase[phasebin_ind, nturn_, :] += L_df.real[ind, :].real
            U_flux_phase[phasebin_ind, nturn_, :] += L_df.imag[ind, :].real
            PA_flux_phase[phasebin_ind, nturn_, :] += PA_df[ind, :]
            V_flux_phase[phasebin_ind, nturn_, :] += V[ind, :]


            counts[phasebin_ind, nturn_, :] += 1
            

        # print("dividing by counts")
        I_flux_phase /= counts
        # print("herei")
        L_flux_phase /= counts
        # print("herei")
        Q_flux_phase /= counts
        # print("herei")
        U_flux_phase /= counts
        # print("herei")
        PA_flux_phase /= counts
        # print("herei")
        V_flux_phase /= counts
        # print("herei")
                
        I_flux_phase[np.isnan(I_flux_phase)] = 0.0
        L_flux_phase[np.isnan(L_flux_phase)] = 0.0
        Q_flux_phase[np.isnan(Q_flux_phase)] = 0.0
        U_flux_phase[np.isnan(U_flux_phase)] = 0.0
        PA_flux_phase[np.isnan(PA_flux_phase)] = 0.0
        V_flux_phase[np.isnan(V_flux_phase)]= 0.0
        
        return(I_flux_phase, L_flux_phase, Q_flux_phase, U_flux_phase, PA_flux_phase, V_flux_phase)
    
    def spidx(self, t, p0=None):
        I, _, _, _ = self.get_stokes()
        It = I[t]
        freqs = self.freqs / 1e9 # frequency in GHz
        
        # remove anyvalue that is lower than 0...
        databool = It > 0.
        It = It[databool]
        freqs = freqs[databool]
        
        logy = np.log(It)
        logx = np.log(freqs)

        try: fval, ferr = model_fit(logx, logy, linearline, p0=p0)
        except: fval = [np.nan]; ferr = [np.nan]
        return fval[0], ferr[0]
    

    
#### function for fitting
from scipy.optimize import curve_fit

def powerlaw(x, A, alpha):
    return A * x ** alpha

def linearline(x, k, b):
    return k * x + b

def model_fit(xdata, ydata, model=linearline, p0=None):
    popt, pcov = curve_fit(
        f=model, xdata=xdata, ydata=ydata, p0=p0
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


