from vis_ds import DynamicSpectra

from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy import units
from astropy.time import Time

from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean

import numpy as np
import pandas as pd

class BurstDS:
    F0 = 0.0003764698195372072743
    
    MeerKAT_location = EarthLocation.of_site("MeerKAT")
    ASKAP_location = EarthLocation.of_site("ASKAP")
    GMRT_location = EarthLocation.of_site("GMRT")
    ATCA_location = EarthLocation.of_address("Narrabri NSW")
    VLA_location = EarthLocation.of_site("VLA")
    
    direction = SkyCoord("18h32m48.41s -09d11m15.82s")
    
    def __init__(
        self, fname,
        period=None,
        pzero=0.1,
        telescope="MKT",
    ):
        if period is None: self.period = 1 / self.F0 / 86400
        else: self.period = period
        self.pzero = pzero
        self.load_ds_file(fname, telescope=telescope)
        ### barycentric the time...
        
    def bary_time(self, telescope):
        if telescope == "MKT":
            ut = Time(self.time, format="mjd", scale="utc", location=self.MeerKAT_location)
        elif telescope == "ASKAP":
            ut = Time(self.time, format="mjd", scale="utc", location=self.ASKAP_location)
        elif telescope == "GMRT":
            ut = Time(self.time, format="mjd", scale="utc", location=self.GMRT_location)
        elif telescope == "ATCA":
            ut = Time(self.time, format="mjd", scale="utc", location=self.ATCA_location)
        elif telescope == "VLA":
            ut = Time(self.time, format="mjd", scale="utc", location=self.VLA_location)
        
        ltt_bary = ut.light_travel_time(
            self.direction, ephemeris='https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de405.bsp'
        )
        batc = ut.tdb + ltt_bary
#         return batc
        self.time = batc.value
            
    
    def load_ds_file(self, fname, telescope):
        if fname.endswith(".npz"):
            self._load_npz(fname)
        elif fname.endswith(".npy"):
            self._load_npy(fname)
        elif fname.endswith(".pkl"):
            self._load_pkl(fname)
        elif fname.endswith(".csv"):
            self._load_csv(fname)
            
        self.bary_time(telescope=telescope)
        ### get lightcurve
        self.lc = np.nanmean(self.data, axis=1)
        self.ftime = (self.time - self.pzero) / self.period
        
        self.cycle = self.ftime // 1.
        self.phase = self.ftime % 1.
    
    def _load_npz(self, fname):
        arr = np.load(fname)
        self.data = arr["stokesI"]
        self.time = arr["time"]
        
    def _load_npy(self, fname):
        self.data = np.load(fname)
        ### get time path
        folder = "/".join(fname.split("/")[:-1])
        self.time = np.load(f"{folder}/time.npy") / 86400
        
        ### sort the data accordingly
        order_index = np.argsort(self.time)
        self.time = self.time[order_index]
        self.data = self.data[order_index]
        
    def _load_pkl(self, fname):
        try:
            ds = DynamicSpectra(fname)
            self.data, _, _, _ = ds.get_stokes()
            self.time = ds.times / 86400.
        except:
            ### use numpy to load it...
            dat = np.load(fname, allow_pickle=True)
            self.data = ((dat["DS"][..., 0] + dat["DS"][..., 1]) / 2).real
            self.time = dat["TIMES"] / 86400.

    def _load_csv(self, fname):
        ### note - this for vlite data only...
        data = pd.read_csv(fname)
        self.data = data[["peak_flux"]].to_numpy()
        self.time = data["mjdtime"].to_numpy()
            
        
    def fold_lc(self, nbins=20, norm=True, removebase=True, pwindow=(0.3, 0.425)):
        bins = np.linspace(0, 1, nbins)
        fold_lc = []; fold_raw_lc = []
        for i in range(nbins-1):
            data = self.lc[(self.phase >= bins[i]) & (self.phase < bins[i+1])]
            fold_raw_lc.append(data)
            if len(data) == 0: fold_lc.append(np.nan)
            else: fold_lc.append(np.nanmean(data))
        self._flc_raw = fold_raw_lc
        self.fphase = (bins[:-1] + bins[1:]) / 2.
        self.flc = np.array(fold_lc)
        
        ### get baselines and remove it
        if removebase:
            self.flc = self._remove_baseline(self.flc, self.fphase, pwindow=pwindow)

        if norm: 
            self.flc = (self.flc - np.nanmin(self.flc)) / (np.nanmax(self.flc) - np.nanmin(self.flc))
        

    def _remove_baseline(self, flc, fphase, pwindow=None):
        if pwindow is None: raise NotImplementedError("a pulse window need to be provided...")
        flcbase = flc.copy()
        flcbase[(fphase % 1 >= pwindow[0]) & (fphase % 1 <= pwindow[1])] = np.nan
        basemean = np.nanmean(flcbase)
        if np.isnan(basemean): basemean = 0.
        return flc - basemean

class BurstPolDS(BurstDS):

    C = 299792458
    
    def load_ds_file(self, fname, telescope):
        if telescope != "ASKAP": raise NotImplementedError("Only ASKAP pkl file allowed for this class...")

        assert fname.endswith(".pkl"), "dynamic spectrum file provided is not a pkl file..."
        self._load_pkl(fname)
        
        self.bary_time(telescope=telescope)
        self.ftime = (self.time - self.pzero) / self.period
        
        self.cycle = self.ftime // 1.
        self.phase = self.ftime % 1.
        
    def _load_pkl(self, fname):
        ### note... for ASKAP only...
        ds = DynamicSpectra(fname, calCASA=False)
        self.I, self.Q, self.U, self.V = ds.get_stokes()
        self.time = ds.times / 86400.
        self.freq = ds.freqs
        self.lambdas = self.C / self.freq
            
        
    def fold_ds(self, nbins=20, ):
        stokesdata = [self.I, self.Q, self.U, self.V]
        foldstokes = [[], [], [], []]

        bins = np.linspace(0, 1, nbins+1)
        for i in range(nbins):
            binbool = (self.phase >= bins[i]) & (self.phase < bins[i+1])
            for idata in range(len(stokesdata)):
                databin = stokesdata[idata][binbool, :]
                if databin.shape[0] == 0: dataave = np.ones((1, len(self.freq))) * np.nan
                else: dataave = np.nanmean(databin, axis=0, keepdims=True)
                foldstokes[idata].append(dataave)
                
        self.fphase = (bins[:-1] + bins[1:]) / 2.
        ### get folded I, Q, U, V
        self.fI = np.concatenate(foldstokes[0], axis=0)
        self.fQ = np.concatenate(foldstokes[1], axis=0)
        self.fU = np.concatenate(foldstokes[2], axis=0)
        self.fV = np.concatenate(foldstokes[3], axis=0)
        
        self.fIlc = np.nanmedian(self.fI, axis=1)
        self.fVlc = np.nanmedian(self.fV, axis=1)
        
    ### rmsynthesise
    def run_rm_synthesize(self, dQs=0.016, dUs=0.016, cleanthres=3, verbose=False):
        rms = []; drms = []; pas = []; dpas = []
        for ibin in range(len(self.fphase)):
            if verbose and (ibin % 10 == 0): print(f"running rm clean on {ibin}/{len(self.fphase)} bin...")
            data = [self.freq, self.fQ[ibin], self.fU[ibin], dQs, dUs]
            mDict, aDict = run_rmsynth(data)
            mDict_cl, aDict_cl = run_rmclean(mDict, aDict, -cleanthres)
            
            RM = mDict_cl["phiPeakPIfit_rm2"]; dRM = mDict_cl["dPhiPeakPIfit_rm2"]
            PA = mDict_cl["polAngle0Fit_deg"]; dPA = mDict_cl["dPolAngle0Fit_deg"]
            
            rms.append(RM); drms.append(dRM)
            pas.append(PA); dpas.append(dPA)
        self.rms = np.array(rms); self.drms = np.array(drms)
        self.pas = np.array(pas); self.dpas = np.array(dpas)
        
    ### defaraday
    def defaraday(self, ):
        fLlc = []
        for ibin in range(len(self.fphase)):
            Q = self.fQ[ibin]; U = self.fU[ibin]
            RM = self.rms[ibin]; PA = np.deg2rad(self.pas[ibin])
            L = Q + 1j * U
            L = np.exp(1j * 2.0 * ( - RM * self.lambdas ** 2.0 - PA)[None, :]) * L
            fLlc.append(L[0].real)
        fLlc = np.array(fLlc)
        self.fLlc = np.nanmedian(fLlc, axis=1)