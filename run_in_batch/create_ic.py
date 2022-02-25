import numpy as np
import sys
import pandas as pd
from astropy import constants
sys.path.insert(1, '../src/')
import integrate_odes as io

outpath = str(sys.argv[1])

Nsim = int(sys.argv[2])
seed = int(sys.argv[3])

np.random.seed(seed)

# Masses and radii
m1, R1 = 1.0, constants.R_sun.to_value("au")
m2, R2 = 5*constants.M_jup.to_value("M_sun"), constants.R_jup.to_value("au")
m3 = 1.0

# Initial eccentricity of inner binary
e1 = 0.01

# Spin periods of the primary and secondary of the inner binary
Pstar1 = 3.0 / 365.25
Pstar2 = 10.0 / (24 * 365.25)

# Internal structure constants
k21, k22 = 1.5 * 0.05, 0.37 / 2
ks1, ks2 = 0.1, 0.25

# Tidal lag time (yr)
tlag1 = 0.0
tlag2 = 1.0 / (3.15e7)

# Magnetic braking coefficients
alpha1 = 1.5e-14
alpha2 = 0.0

for simnum in range(1,Nsim + 1):

    a1 = np.random.uniform(1, 5)
    a2 = np.random.uniform(100, 1000)
    e2 = np.random.uniform(0.001, 0.9)

    Imut = np.arccos(np.random.uniform(-1,1))

    I1, I2 = io.set_inclinations(m1, m2, m3, a1, e1, a2, e2, Imut)

    peri1, node1 = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)
    peri2 = np.random.uniform(0, 2 * np.pi)

    node2 = node1 - np.pi

    # Evaluate stability condition
    mfac = (1 + m3/(m1 + m2)) ** (2.0 / 5.0)
    efac = ((1 + e2) ** (2.0 / 5.0)) / ((1 - e2) ** (6.0 / 5.0))
    stable = 2.8*mfac*efac*(1 - 0.3 * Imut / np.pi)

    if a2/a1 > stable:

        # Create a dictionary to store the initial conditions and triple properties
        parameters = {
            "simnum": simnum,
            "seed":seed,
            "m1": m1,
            "m2": m2,
            "m3": m3,
            "R1": R1,
            "R2": R2,
            "a1": a1,
            "a2": a2,
            "e1": e1,
            "e2": e2,
            "k21": k21,
            "k22": k22,
            "ks1": ks1,
            "ks2": ks2,
            "tlag1": tlag1,
            "tlag2": tlag2,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "Pstar1": Pstar1,
            "Pstar2": Pstar2,
            "I1": I1,
            "node1": node1,
            "peri1": peri1,
            "I2": I2,
            "node2": node2,
            "peri2": peri2,
            "Imom1": ks1 * m1 * R1 * R1,
            "Imom2": ks2 * m2 * R2 * R2,
            "Imut":Imut
        }

        df = pd.DataFrame(parameters, index=[0])

        # Save file
        outfile = outpath + "ic" + str(simnum) + ".csv"
        df.to_csv(outfile, index=False)








