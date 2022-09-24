import numpy as np
import astropy.units as u
from scipy.optimize import fsolve

# Julian Date of Vernal Equinox at 2000: 2451623.815972
# Julian Date of Perihelion at 2000 : 2451546.708333
# Earth Orbital Period : 365.25 days
# Earth Obliquity : 23.44 deg
# Orbit eccentricity : 0.0167
# date = JD - 2450000

TE = 1623.815972 #day
TPeri = 1546.708333
P = (1 * u.year).to(u.day).value
E = (23.44 * u.degree).to(u.radian).value
e = 0.0167
sec = (1 * u.day).to(u.second).value

def psi_t(t, T0=0):
    theta = ((t - T0) / P - np.floor((t - T0) / P)) * np.pi * 2
    def Psi(psi):
        return psi - e * np.sin(psi) - theta

    return fsolve(Psi, [theta])


def Earth_to_Sun(HJD):
    # Eccentric anomaly of the Vernal Equinox at 2000
    xE = (np.cos(psi_t(TE, T0=TPeri)) - e)
    yE = (np.sqrt(1 - e**2) * np.sin(psi_t(TE, T0=TPeri)))
    Delta = np.arctan2(yE, xE)
    X = (np.cos(psi_t(HJD, T0=TPeri)) - e)
    Y = (np.sqrt(1 - e**2) * np.sin(psi_t(HJD, T0=TPeri)))
    XE = (X * np.cos(Delta) + Y * np.sin(Delta))
    YE = (-X * np.sin(Delta) + Y * np.cos(Delta)) * np.cos(E)
    ZE = (-X * np.sin(Delta) + Y * np.cos(Delta)) * np.sin(E)
    return (XE, YE, ZE)

class Lensing:
    def __init__(self,
                 T,
                 RA,
                 Dec,
                 Ms=None,
                 Mb=None,
                 Fs=None,
                 Fb=None,
                 T0=None,
                 TP=None,
                 u0=None,
                 tE=None,
                 PiEN=None,
                 PiEE=None,
                 Best=[]):
        '''
        Initialization

        Parameters
        ------------------------------
        T: quantity, in astropy.units.day
            JD - 2450000 in the earth frame
        RA, Dec: float
            Coordinates in degrees
        Ms, Mb: float
            Magnitudes of the source/blending
        Fs, Fb: float
            Fluxes (arbitrary units) of the source/blending
        T0: quantity, in astropy.units.day
            t0 - the MJD at the closest encounter
        TP: quantity, in astropy.units.day
            the reference time to calculate the velocity
            of the Sun's motion
        u0: float
            mu0 - impact parameter
        tE: quantity, in astropy.units.day
            tE - Einstein timescale
        PiEN, PiEE: float
            the north/east components of the microlensing parallax
        '''
        if len(Best) > 0:
            T0, u0, tE, PiEN, PiEE = Best
        if TP == None:
            TP = T0
        self.RA, self.Dec = (RA * u.degree).to(u.radian).value, (Dec * u.degree).to(u.radian).value
        self.T, self.T0, self.TP = T, T0, TP
        self.tE = tE
        self.u0 = u0
        self.PiEN, self.PiEE = PiEN, PiEE
        self.Ms, self.Mb = Ms, Mb
        self.Fs, self.Fb = Fs, Fb
        self.A()

    def ds(self, T):
        '''
        calculate delta s in Gould 2004

        Parameters
        -------------------------------
        T: float or array-like (quantity, in astropy.units.day)
            JD - 2450000 in the earth frame
        '''
        EtS = np.array(Earth_to_Sun(T))
        vec = np.array([
            np.cos(self.Dec) * np.cos(self.RA),
            np.cos(self.Dec) * np.sin(self.RA),
            np.sin(self.Dec)
        ])
        vecN = np.array([0, 0, 1])
        vecE = np.cross(vecN, vec)
        vecN = np.cross(vec, vecE)
        vecN = vecN / np.sqrt(np.dot(vecN, vecN))
        vecE = vecE / np.sqrt(np.dot(vecE, vecE))
        projN = [np.dot(EtS[:, i], vecN) for i in range(len(EtS[0, :]))]
        projE = [np.dot(EtS[:, i], vecE) for i in range(len(EtS[0, :]))]

        Tp = self.TP
        EtS0 = np.array(Earth_to_Sun(Tp)).flatten()
        projN0 = np.dot(EtS0, vecN)
        projE0 = np.dot(EtS0, vecE)
        EtS1 = np.array(Earth_to_Sun(Tp - 1)).flatten()
        EtS2 = np.array(Earth_to_Sun(Tp + 1)).flatten()
        vN = np.dot(EtS2 - EtS1, vecN) / 2
        vE = np.dot(EtS2 - EtS1, vecE) / 2
        dsN, dsE = projN - projN0 - vN * \
            (T - Tp), projE - projE0 - vE * (T - Tp)
        self.acc = [projN0, projE0]
        self.vel = [vN, vE]
        return (dsN, dsE)

    def A(self):
        '''
        Get magnification - self.A
        '''
        self.dsN, self.dsE = self.ds(self.T)
        tau = (self.dsN * self.PiEN + self.dsE * self.PiEE) + \
            (self.T - self.T0) / self.tE
        beta = (self.dsE * self.PiEN - self.dsN * self.PiEE) + self.u0
        u2 = tau**2 + beta**2
        tau0 = (self.T - self.T0) / self.tE
        beta0 = self.u0
        u02 = tau0**2 + beta0**2
        A = (u2 + 2) / np.sqrt(u2**2 + 4 * u2)
        A0 = (u02 + 2) / np.sqrt(u02**2 + 4 * u02)
        if self.Fs == None or self.Fb == None:
            fm = 10**(-0.4 * self.Ms) * A + 10**(-0.4 * self.Mb)
            fm0 = 10**(-0.4 * self.Ms) * A0 + 10**(-0.4 * self.Mb)
            self.M, self.M0 = -2.5 * np.log10(fm), -2.5 * np.log10(fm0)
        else:
            fm = self.Fs * A + self.Fb
            fm0 = self.Fs * A0 + self.Fb
            self.M, self.M0 = -2.5 * \
                np.log10(fm) + 18, -2.5 * np.log10(fm0) + 18

    def predict(self, T):
        '''
        Predict the flux at any given time

        Parameters
        -------------------------------
        T: float or array-like (quantity, in astropy.units.day)
            MJD at the observation
        '''
        dsN, dsE = self.ds(T)
        tau = (dsN * self.PiEN + dsE * self.PiEE) + \
            (T - self.T0) / self.tE
        beta = (dsE * self.PiEN - dsN * self.PiEE) + self.u0
        u2 = tau**2 + beta**2
        A = (u2 + 2) / np.sqrt(u2**2 + 4 * u2)
        if self.Fs == None or self.Fb == None:
            fm = 10**(-0.4 * self.Ms) * A + 10**(-0.4 * self.Mb)
            M = -2.5 * np.log10(fm)
        else:
            fm = self.Fs * A + self.Fb
            M = -2.5 * np.log10(fm) + 18
        return fm
