from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy import units

import numpy as np

class BurstEphemeris:
    def __init__(
        self, pzero, period, pcenter, pwidth, 
        coord = "18h32m48.41s -09d11m15.82s",
        site = "ASKAP",
    ):
        self.pzero = pzero
        self.period = period
        self.pcenter = pcenter
        self.pwidth = pwidth
        self.coord = SkyCoord(coord)
        self.siteloc = self._parse_site(site)

    def _parse_site(self, site):
        if site == "ASKAP":
            ASKAP_longitude = Angle("116:38:13.0", unit=units.deg)
            ASKAP_latitude = Angle("-26:41:46.0", unit=units.deg)
            return EarthLocation(lat=ASKAP_latitude, lon=ASKAP_longitude)
        if site == "MeerKAT":
            MeerKAT_longitude = Angle("21:19:48.0", unit=units.deg)
            MeerKAT_latitude = Angle("-30:49:48.0", unit=units.deg)
            return EarthLocation(lat=MeerKAT_latitude, lon=MeerKAT_longitude)
        if site == "ATCA":
            ATCA_longitude = Angle(149.5619665, unit=units.degree)
            ATCA_latitude = Angle(-30.3138992, unit=units.degree)
            return EarthLocation(lon=ATCA_longitude, lat=ATCA_latitude)
        try:
            return EarthLocation.of_site(site)
        except:
            return EarthLocation.of_address(site)
        raise ValueError(f"{site} is not in the list...")
    
    def bary_time(self, topomjd, scale="utc"):
        topot = Time(topomjd, format="mjd", scale=scale, location=self.siteloc)
        ltt_bary = topot.light_travel_time(self.coord, ephemeris='jpl')
        return topot.tdb + ltt_bary
    
    def topo_time(self, barymjd, scale="tdb"):
        baryt = Time(barymjd, format="mjd", scale=scale, location=self.siteloc)
        ltt_topo = baryt.light_travel_time(self.coord, ephemeris="jpl")
        return baryt.utc - ltt_topo
    

    def get_pulse_range(self, tstart, tend):
        """
        use tstart and tend to determine the pulse ranges...
        tstart and tend both in mjds, not barycentric...
        """
        barystart = self.bary_time(tstart, ).value
        baryend = self.bary_time(tend, ).value
        nstart = int(np.floor((barystart - self.pzero) / self.period))
        nend = int(np.ceil((baryend - self.pzero) / self.period))

        bary_tranges =  [
            (
                self.pzero + (i + (self.pcenter - self.pwidth/2)) * self.period,
                self.pzero + (i + (self.pcenter + self.pwidth/2)) * self.period,
            )
            for i in range(nstart, nend)
        ]

        ### convert barycentric time back to topocentric...
        return [
            (self.topo_time(tstart).value, self.topo_time(tend).value)
            for tstart, tend in bary_tranges
        ]
        
    def _format_casa_time(self, mjd):
        t = Time(mjd, format="mjd")
        return t.strftime('%Y/%m/%d/%H:%M:%S.%f')
    
    def format_casa_trange(self, tstart, tend):
        """
        follow casa convention, print out the time range
        """
        return f"{self._format_casa_time(tstart)}~{self._format_casa_time(tend)}"

def main(time, telescope):
    pzero = 0.
    period = 0.03074369703340903 #day
    pcenter = 0.6
    pwidth = 0.075

    burstephemeris = BurstEphemeris(
        pzero=pzero, period=period,
        pcenter=pcenter, pwidth=pwidth,
        site=telescope,
    )

    tstart = Time(f"{time} 00:00:00", format="iso").mjd
    tend = Time(f"{time} 23:59:59", format="iso").mjd

    tranges = burstephemeris.get_pulse_range(tstart, tend)
    for trange in tranges:
        print(burstephemeris.format_casa_trange(*trange))


if __name__ == "__main__":
    ### get time for a pulses for a given day - in UTC
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Get pulses arrival time from J1832', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--date", type=str, help="date in UTC to check, in the format of YYYY-MM-DD", )
    parser.add_argument("-t", "--telescope", type=str, help="telescope", default="MeerKAT")
    values = parser.parse_args()

    main(values.date, values.telescope)




    
