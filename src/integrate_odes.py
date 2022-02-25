import numpy as np
import scipy.integrate as integ
import pandas as pd
from astropy import constants


# Gravitational constant
G = constants.G.to_value("au^3/yr^2 M_sun")
# Speed of light in au/yr
clight = constants.c.to_value("au/yr")

#Notes
# y_func(), integration_loop(), and integrate_triple() are functions used to solve the system of ODEs
#set_inclinations(), set_triple_properties() are functions to set up the simulation
# process_output(), calc_a_e(), and calc_features() are functions to process the ODE solution

def y_func(t, y, p):

    """Calculates the derivatives for the angular momentum, eccentricity, and spin vectors, to use in the ODE integration. Takes a time t, vector of dependent variables y, and tuple of parameters p."""

    (
        m1,
        m2,
        m3,
        R1,
        R2,
        k21,
        k22,
        ks1,
        ks2,
        tlag1,
        tlag2,
        alpha1,
        alpha2,
        perturber,
        gr_peri,
        tide1_peri,
        tide2_peri,
        spin1_peri,
        spin2_peri,
        evolve_spin2_axis,
        spin1_node,
        spin2_node,
        diss_tide1,
        diss_tide2,
        oct,
        mbraking1,
        mbraking2,
    ) = p

    kq1, kq2 = (2.0 / 3.0) * k21, (2.0 / 3.0) * k22
    klove1, klove2 = 2 * k21, 2 * k22

    m12 = m1 + m2
    m123 = m12 + m3
    mu_in, mu_out = m1 * m2 / m12, m12 * m3 / m123

    L1x, L1y, L1z = y[0], y[1], y[2]
    e1x, e1y, e1z = y[3], y[4], y[5]
    L2x, L2y, L2z = y[6], y[7], y[8]
    e2x, e2y, e2z = y[9], y[10], y[11]
    S1x, S1y, S1z = y[12], y[13], y[14]

    L1 = np.sqrt(L1x**2 + L1y**2 + L1z**2)
    L2 = np.sqrt(L2x**2 + L2y**2 + L2z**2)
    e1 = np.sqrt(e1x**2 + e1y**2 + e1z**2)
    e2 = np.sqrt(e2x**2 + e2y**2 + e2z**2)
    S1 = np.sqrt(S1x**2 + S1y**2 + S1z**2)

    # Unit vectors
    l1x, l1y, l1z = L1x / L1, L1y / L1, L1z / L1
    l2x, l2y, l2z = L2x / L2, L2y / L2, L2z / L2
    s1x, s1y, s1z = S1x / S1, S1y / S1, S1z / S1

    l1hat = np.array([l1x, l1y, l1z])
    e1hat = np.array([e1x / e1, e1y / e1, e1z / e1])
    l2hat = np.array([l2x, l2y, l2z])
    e2hat = np.array([e2x / e2, e2y / e2, e2z / e2])
    s1hat = np.array([s1x, s1y, s1z])

    e1vec = np.array([e1x, e1y, e1z])
    e2vec = np.array([e2x, e2y, e2z])

    e1_sq = e1**2
    e2_sq = e2**2

    j1_sq = 1 - e1_sq
    j2_sq = 1 - e2_sq
    j1, j2 = np.sqrt(j1_sq), np.sqrt(j2_sq)
    j1_3 = j1**3
    j1_13 = j1**13

    j1vec = j1 * l1hat

    if evolve_spin2_axis == 1:
        S2x, S2y, S2z = y[15], y[16], y[17]
        S2 = np.sqrt(S2x**2 + S2y**2 + S2z**2)
        s2x, s2y, s2z = S2x / S2, S2y / S2, S2z / S2
        s2hat = np.array([s2x, s2y, s2z])

    else:
        # Evolve spin magnitude only and force s2hat to be along orbital axis
        S2 = y[15]
        s2hat = np.array([l1x, l1y, l1z])

    L10, L20 = L1 / j1, L2 / j2

    a1 = (L1**2) / (mu_in * mu_in * G * m12 * j1_sq)
    a2 = (L2**2) / (mu_out * mu_out * G * m123 * j2_sq)
    a2eff = a2 * j2
    n1, n2 = np.sqrt(G * m12 / (a1**3)), np.sqrt(G * m123 / (a2**3))
    Omega1, Omega2 = S1 / (ks1 * m1 * R1 * R1), S2 / (ks2 * m2 * R2 * R2)
    Omega1b, Omega2b = np.sqrt(G * m1 / (R1**3)), np.sqrt(G * m2 / (R2**3))
    Omega_hat1, Omega_hat2 = Omega1 / Omega1b, Omega2 / Omega2b

    Omega1_over_n1 = Omega1 / n1
    Omega2_over_n1 = Omega2 / n1

    eps_oct = ((m1 - m2) / m12) * (a1 / a2) * e2 / j2_sq

    ####################################

    s1hat_dot_l1hat = np.dot(s1hat, l1hat)
    s2hat_dot_l1hat = np.dot(s2hat, l1hat)
    s1hat_cross_l1hat = np.cross(s1hat, l1hat)
    s2hat_cross_l1hat = np.cross(s2hat, l1hat)
    s1hat_cross_e1 = np.cross(s1hat, e1vec)
    s2hat_cross_e1 = np.cross(s2hat, e1vec)
    e1_dot_s1hat = np.dot(e1vec, s1hat)
    e1_dot_s2hat = np.dot(e1vec, s2hat)

    l1hat_cross_e1 = np.cross(l1hat, e1vec)

    e1_dot_e2 = np.dot(e1vec, e2vec)
    e2_cross_e1 = np.cross(e2vec, e1vec)
    e1_dot_l2hat = np.dot(e1vec, l2hat)
    e1_cross_l2hat = np.cross(e1vec, l2hat)
    l2hat_cross_e2 = np.cross(l2hat, e2vec)

    j1_cross_e1 = np.cross(j1vec, e1vec)
    j1_cross_e2 = np.cross(j1vec, e2vec)
    j1_dot_l2hat = np.dot(j1vec, l2hat)
    j1_cross_l2hat = np.cross(j1vec, l2hat)
    j1_dot_e2hat = np.dot(j1vec, e2hat)
    j1_dot_e2 = np.dot(j1vec, e2vec)
    j1_cross_e2hat = np.cross(j1vec, e2hat)

    e1_cross_e2hat = np.cross(e1vec, e2hat)
    e1_dot_e2hat = np.dot(e1vec, e2hat)
    e2hat_cross_j1 = np.cross(e2hat, j1vec)
    e2hat_cross_e1 = np.cross(e2hat, e1vec)
    l2hat_cross_e1 = np.cross(l2hat, e1vec)
    e2_cross_l2hat = np.cross(e2vec, l2hat)
    l2hat_cross_j1 = np.cross(l2hat, j1vec)

    cos_theta1 = s1hat_dot_l1hat
    cos_theta2 = s2hat_dot_l1hat

    g = (1 + 1.5 * e1_sq + (e1_sq**2) / 8) / (j1**10)

    tk = (m12 / m3) * ((a2eff / a1) ** 3) * (1.0 / n1)
    wdot_gr = 3 * G * m12 * n1 / ((clight**2) * a1 * j1_sq)

    wdot_tide1 = 15 * n1 * (m2 / m1) * k21 * g * (R1 / a1) ** 5
    wdot_tide2 = 15 * n1 * (m1 / m2) * k22 * g * (R2 / a1) ** 5

    wdot_rot1 = 1.5 * kq1 * ((R1 / a1) ** 2) * (Omega_hat1**2) * n1 / (j1**4)
    wdot_rot2 = 1.5 * kq2 * ((R2 / a1) ** 2) * (Omega_hat2**2) * n1 / (j1**4)

    # LK equations for inner and outer orbits

    # Quadrupole terms
    dL1_dt_quad = (
        (3.0 / (4.0 * tk * j1))
        * L1
        * (j1_dot_l2hat * j1_cross_l2hat - 5 * e1_dot_l2hat * e1_cross_l2hat)
    )
    de1_dt_quad = (3.0 / (4.0 * tk)) * (
        j1_dot_l2hat * e1_cross_l2hat
        + 2 * j1_cross_e1
        - 5 * e1_dot_l2hat * j1_cross_l2hat
    )

    dL2_dt_quad = (
        (3.0 / (4.0 * tk))
        * (L2 / j2)
        * (L10 / L20)
        * (-j1_dot_l2hat * j1_cross_l2hat + 5 * e1_dot_l2hat * e1_cross_l2hat)
    )
    de2_dt_quad = (
        (3.0 / (4.0 * tk * j2))
        * (L10 / L20)
        * (
            -j1_dot_l2hat * j1_cross_e2
            - 5 * e1_dot_l2hat * e2_cross_e1
            - (
                0.5
                - 3 * e1_sq
                + (25.0 / 2.0) * e1_dot_l2hat**2
                - (5.0 / 2.0) * j1_dot_l2hat**2
            )
            * l2hat_cross_e2
        )
    )

    # Octupole terms

    # dL1/dt
    C = (8.0 / 5.0) * e1_sq - (1.0 / 5.0) - 7 * (e1_dot_l2hat**2) + j1_dot_l2hat**2

    term1 = (
        2 * (e1_dot_l2hat * j1_dot_l2hat + e1_dot_l2hat * j1_dot_e2hat) * j1_cross_l2hat
        + 2
        * (j1_dot_e2hat * j1_dot_l2hat - 7 * e1_dot_e2hat * e1_dot_l2hat)
        * e1_cross_l2hat
    )

    term2 = 2 * e1_dot_l2hat * j1_dot_l2hat * j1_cross_e2hat + C * e1_cross_e2hat

    dL1_dt_oct = -(75.0 / 64.0) * (L1 / (tk * j1)) * eps_oct * (term1 + term2)

    # de1/dt
    term1 = 2 * e1_dot_l2hat * j1_dot_l2hat * e1_cross_e2hat + C * j1_cross_e2hat

    term2 = (
        2 * (e1_dot_e2hat * j1_dot_l2hat + e1_dot_l2hat * j1_dot_e2hat) * e1_cross_l2hat
        + 2
        * (j1_dot_l2hat * j1_dot_e2hat - 7 * e1_dot_l2hat * e1_dot_e2hat)
        * j1_cross_l2hat
    )

    term3 = (16.0 / 5.0) * e1_dot_e2hat * j1_cross_e1

    de1_dt_oct = -(75.0 / 64.0) * (eps_oct / tk) * (term1 + term2 + term3)

    # dL2/dt
    term1 = 2 * (
        e1_dot_l2hat * j1_dot_e2hat * l2hat_cross_j1
        + e1_dot_e2hat * j1_dot_l2hat * l2hat_cross_j1
        + e1_dot_l2hat * j1_dot_l2hat * e2hat_cross_j1
    )
    term2 = (
        2 * j1_dot_e2hat * j1_dot_l2hat * l2hat_cross_e1
        - 14 * e1_dot_e2hat * e1_dot_l2hat * l2hat_cross_e1
        + C * e2hat_cross_e1
    )

    dL2_dt_oct = (
        -(75.0 / (64.0 * tk)) * (L2 / j2) * eps_oct * (L10 / L20) * (term1 + term2)
    )

    # de2/dt
    term1 = 2 * (
        e1_dot_l2hat * j1_dot_e2 * e2hat_cross_j1
        + j1_dot_l2hat * e1_dot_e2 * e2hat_cross_j1
        + (j2_sq / e2) * e1_dot_l2hat * j1_dot_l2hat * l2hat_cross_j1
    )
    term2 = (
        2 * j1_dot_e2 * j1_dot_l2hat * e2hat_cross_e1
        - 14 * e1_dot_e2 * e1_dot_l2hat * e2hat_cross_e1
        + (j2_sq / e2) * C * l2hat_cross_e1
    )
    term3 = -(
        2 * (1.0 / 5.0 - 8.0 * e1_sq / 5.0) * e1_dot_e2hat * e2_cross_l2hat
        + 14 * e1_dot_l2hat * j1_dot_e2hat * j1_dot_l2hat * e2_cross_l2hat
        + 7 * e1_dot_e2hat * C * e2_cross_l2hat
    )

    de2_dt_oct = (
        -(75.0 / 64.0) * (eps_oct / (tk * j2)) * (L10 / L20) * (term1 + term2 + term3)
    )

    ##########################################

    # Precession of the spin axes
    Omega_s1 = (
        1.5 * (kq1 / ks1) * (m2 / m1) * ((R1 / a1) ** 3) * Omega1 / (j1 ** (3.0 / 2.0))
    )
    Omega_s2 = (
        1.5 * (kq2 / ks2) * (m1 / m2) * ((R2 / a1) ** 3) * Omega2 / (j1 ** (3.0 / 2.0))
    )

    dS1_dt_rot = Omega_s1 * S1 * s1hat_dot_l1hat * s1hat_cross_l1hat
    dS2_dt_rot = Omega_s2 * S2 * s2hat_dot_l1hat * s2hat_cross_l1hat

    # Nodal precession due to stellar quadrupoles
    dL1_dt_rot1 = -dS1_dt_rot
    dL1_dt_rot2 = -dS2_dt_rot

    # Precession of the eccentricity vector
    de1_dt_gr = wdot_gr * l1hat_cross_e1
    de1_dt_tide1 = wdot_tide1 * l1hat_cross_e1
    de1_dt_tide2 = wdot_tide2 * l1hat_cross_e1

    de1_dt_rot1 = wdot_rot1 * (
        0.5 * (5 * cos_theta1**2 - 1) * l1hat_cross_e1 - cos_theta1 * s1hat_cross_e1
    )
    de1_dt_rot2 = wdot_rot2 * (
        0.5 * (5 * cos_theta2**2 - 1) * l1hat_cross_e1 - cos_theta2 * s2hat_cross_e1
    )

    # Tidal dissipation
    ta_inv1 = 6 * klove1 * tlag1 * (m2 / m1) * ((R1 / a1) ** 5) * n1 * n1
    ta_inv2 = 6 * klove2 * tlag2 * (m1 / m2) * ((R2 / a1) ** 5) * n1 * n1

    f2 = (
        1 + (15.0 / 2.0) * e1 * e1 + (45.0 / 8.0) * (e1**4) + (5.0 / 16.0) * (e1**6)
    )

    f3 = (
        1 + (15.0 / 4.0) * e1 * e1 + (15.0 / 8.0) * (e1**4) + (5.0 / 64.0) * (e1**6)
    )

    f4 = 1 + (3.0 / 2.0) * e1 * e1 + (e1**4) / 8

    f5 = 1 + 3 * e1 * e1 + (3.0 / 8.0) * (e1**4)

    dS1_dt_tide = (
        -0.5
        * ta_inv1
        * (L1 / (j1_13))
        * (j1_3 * f5 * 0.5 * (s1hat + cos_theta1 * l1hat) * Omega1_over_n1 - f2 * l1hat)

    )
    dL1_dt_tide1 = -dS1_dt_tide
    de1_dt_tide_diss1 = (
        -0.5
        * ta_inv1
        * (1.0 / j1_13)
        * (
            j1_3 * f4 * 0.5 * Omega1_over_n1 * e1_dot_s1hat * l1hat
            - ((11.0 / 2.0) * j1_3 * f4 * Omega1_over_n1 - 9 * f3) * e1vec
        )
    )
    de1_dt_tide_diss2 = (
        -0.5
        * ta_inv2
        * (1.0 / j1_13)
        * (
            j1_3 * f4 * 0.5 * Omega2_over_n1 * e1_dot_s2hat * l1hat
            - ((11.0 / 2.0) * j1_3 * f4 * Omega2_over_n1 - 9 * f3) * e1vec
        )
    )

    # Magnetic braking
    dS1_dt_mb = -alpha1 * S1 * Omega1 * Omega1 * s1hat

    # Full equations of motion

    dL2_dt = dL2_dt_quad * perturber + dL2_dt_oct * oct * perturber
    de2_dt = de2_dt_quad * perturber + de2_dt_oct * oct * perturber

    dS1_dt = dS1_dt_rot + dS1_dt_tide * diss_tide1 + dS1_dt_mb * mbraking1

    if evolve_spin2_axis == 1:

        dS2_dt_tide = (
            -0.5
            * ta_inv2
            * (L1 / (j1_13))
            * (
                (j1_3) * f5 * 0.5 * (s2hat + cos_theta2 * l1hat) * Omega2_over_n1
                - f2 * l1hat
            )
        )
        dS2_dt_mb = -alpha2 * S2 * Omega2 * Omega2 * s2hat
        dL1_dt_tide2 = -dS2_dt_tide
        dS2_dt = dS2_dt_rot + dS2_dt_tide * diss_tide2 + dS2_dt_mb * mbraking2

        dydt = np.empty(18, float)
        dydt[15] = dS2_dt[0]
        dydt[16] = dS2_dt[1]
        dydt[17] = dS2_dt[2]
    else:

        dS2mag_dt_tide = (
            -0.5
            * ta_inv2
            * (L1 / j1_13)
            * (j1_3 * f5 * Omega2_over_n1 - f2)
        )
        dS2mag_dt_mb = -alpha2 * S2 * Omega2 * Omega2

        dS2mag_dt = dS2mag_dt_tide * diss_tide2 + dS2mag_dt_mb * mbraking2

        dL1_dt_tide2 = -dS2mag_dt_tide * l1hat

        dydt = np.empty(16, float)
        dydt[15] = dS2mag_dt

    de1_dt = (
        de1_dt_quad * perturber
        + de1_dt_oct * oct * perturber
        + de1_dt_gr * gr_peri
        + de1_dt_tide1 * tide1_peri
        + de1_dt_tide2 * tide2_peri
        + de1_dt_rot1 * spin1_peri
        + de1_dt_rot2 * spin2_peri
        + de1_dt_tide_diss1 * diss_tide1
        + de1_dt_tide_diss2 * diss_tide2
    )

    dL1_dt = (
        dL1_dt_quad * perturber
        + dL1_dt_oct * oct * perturber
        + dL1_dt_rot1 * spin1_node
        + dL1_dt_rot2 * evolve_spin2_axis * spin2_node
        + dL1_dt_tide1 * diss_tide1
        + dL1_dt_tide2 * diss_tide2
    )

    dydt[0] = dL1_dt[0]
    dydt[1] = dL1_dt[1]
    dydt[2] = dL1_dt[2]

    dydt[3] = de1_dt[0]
    dydt[4] = de1_dt[1]
    dydt[5] = de1_dt[2]

    dydt[6] = dL2_dt[0]
    dydt[7] = dL2_dt[1]
    dydt[8] = dL2_dt[2]

    dydt[9] = de2_dt[0]
    dydt[10] = de2_dt[1]
    dydt[11] = de2_dt[2]

    dydt[12] = dS1_dt[0]
    dydt[13] = dS1_dt[1]
    dydt[14] = dS1_dt[2]

    return dydt


def integration_loop(
    d,
    p,
    y0,
    t,
    sol,
    times,
    a,
    e,
    peri,
    eps_gr,
    min_a,
    min_e,
    min_peri,
    eps_gr_max,
    enforce_a_e_condition=False,
    enforce_epsgr_condition=False,
    meth="bdf",
    err=1e-9,
    resume=False,
):

    """Function to perform numerical integration of the ODEs, looping over time array t and checking for various stopping conditions. Returns ODE solution and flags specifying the outcome of the numerical integration."""

    t0 = t[0]
    tfinal = t[-1]

    if meth == "bdf" or meth == "adams":
        solver = integ.ode(y_func).set_integrator(
            "vode", method=meth, atol=err, rtol=err, nsteps=50000
        )
    elif meth == "dop853" or meth == "lsoda":
        solver = integ.ode(y_func).set_integrator(
            meth, atol=err, rtol=err, nsteps=50000
        )

    solver.set_initial_value(y0, t0).set_f_params(p)

    i = 0
    current_t = t[i]

    # Integration loop
    while solver.successful() and i < len(t) - 1 and peri > min_peri:

        i += 1
        current_t = t[i]

        solver.integrate(current_t)

        if solver.successful():
            times.append(current_t)
            sol.append(solver.y)

        a, e, peri = calc_a_e(solver.y, d["m1"], d["m2"])
        d_features = calc_features(a, np.nan, np.nan, d["e2"], d)
        eps_gr = d_features["eps_gr"]

        if enforce_a_e_condition:
            if a < min_a and e < min_e:
                break

        if enforce_epsgr_condition:
            if eps_gr > eps_gr_max:
                break

    # Quality flag to determine if integration was successful
    qflag = solver.successful()

    return current_t, solver.y, times, sol, a, e, peri, eps_gr, qflag


def integrate_triple(
    d,
    t,
    perturber=True,
    gr_peri=True,
    tide1_peri=True,
    tide2_peri=True,
    spin1_peri=True,
    spin2_peri=True,
    evolve_spin2_axis=True,
    spin1_node=True,
    spin2_node=True,
    diss_tide1=False,
    diss_tide2=False,
    oct=True,
    mbraking1=False,
    mbraking2=False,
    min_peri=-1,
    min_a=-1,
    min_e=-1,
    max_eps_gr=30,
):

    """Takes the initial conditions and input parameters in the form of a dictionary (d), and integrates the ODEs. Returns the ODE solution, along with flags for the outcome."""

    print("Setting up simulation #", d["simnum"])
    print("GR apsidal precession: ", gr_peri)
    print("Tidal apsidal precession due to primary: ", tide1_peri)
    print("Tidal apsidal precession due to secondary: ", tide2_peri)
    print("Spin apsidal precession due to primary: ", spin1_peri)
    print("Spin apsidal precession due to secondary: ", spin2_peri)
    print("Spin nodal precession due to primary: ", spin1_node)
    print("Spin nodal precession due to secondary: ", spin2_node)
    print("Evolve spin axis of secondary: ", evolve_spin2_axis)
    print("Tidal dissipation in primary: ", diss_tide1)
    print("Tidal dissipation in secondary: ", diss_tide2)
    print("Magnetic braking in primary: ", mbraking1)
    print("Magnetic braking in secondary: ", mbraking2)
    print("Octupole: ", oct)
    print("Minimum pericenter: ", min_peri)
    print("Minimum a and e: ", min_a, min_e)

    # Put the initial conditions in an array to pass to the integration function

    if evolve_spin2_axis == True:
        y0 = [
            d["L1x"],
            d["L1y"],
            d["L1z"],
            d["e1x"],
            d["e1y"],
            d["e1z"],
            d["L2x"],
            d["L2y"],
            d["L2z"],
            d["e2x"],
            d["e2y"],
            d["e2z"],
            d["S1x"],
            d["S1y"],
            d["S1z"],
            d["S2x"],
            d["S2y"],
            d["S2z"],
        ]
    else:
        y0 = [
            d["L1x"],
            d["L1y"],
            d["L1z"],
            d["e1x"],
            d["e1y"],
            d["e1z"],
            d["L2x"],
            d["L2y"],
            d["L2z"],
            d["e2x"],
            d["e2y"],
            d["e2z"],
            d["S1x"],
            d["S1y"],
            d["S1z"],
            d["S2"],
        ]

    # Define a parameter tuple for ODE (because scipy ode routine does not allow a dictionary)
    p = (
        d["m1"],
        d["m2"],
        d["m3"],
        d["R1"],
        d["R2"],
        d["k21"],
        d["k22"],
        d["ks1"],
        d["ks2"],
        d["tlag1"],
        d["tlag2"],
        d["alpha1"],
        d["alpha2"],
        int(perturber),
        int(gr_peri),
        int(tide1_peri),
        int(tide2_peri),
        int(spin1_peri),
        int(spin2_peri),
        int(evolve_spin2_axis),
        int(spin1_node),
        int(spin2_node),
        int(diss_tide1),
        int(diss_tide2),
        int(oct),
        int(mbraking1),
        int(mbraking2),
    )

    y0 = np.asarray(y0)
    sol = [y0]
    times = [t[0]]

    # Make sure the integration loop will run at least once
    peri, a, e = min_peri + 1, min_a + 1, min_e + 1
    eps_gr = max_eps_gr - 1

    current_t, current_y, times, sol, a, e, peri, eps_gr, qflag = integration_loop(
        d,
        p,
        y0,
        t,
        sol,
        times,
        a,
        e,
        peri,
        eps_gr,
        min_a,
        min_e,
        min_peri,
        max_eps_gr,
        False,
        True,
        meth="bdf",
        err=1e-9,
        resume=False,
    )

    print(
        "Exited the first integration loop. t/tmax, a, e, peri, eps_gr = ",
        np.round(current_t/t[-1],2),
        np.round(a, 2),
        np.round(e, 2),
        np.round(peri, 2),
        np.round(eps_gr, 2),
    )

    if (
        current_t < t[-1]
        and peri > min_peri
        and (a < min_a or e < min_e or eps_gr > max_eps_gr)
    ):
        print(
            "Migrating body is de-coupled from tertiary. Ignoring e-precession and tertiary"
        )
        t_new = [ti for ti in list(t) if ti not in times]

        y_new = np.asarray(current_y)

        gr_peri = False
        tide1_peri = False
        tide2_peri = False
        spin1_peri = False
        spin2_peri = False
        perturber = False

        p = (
            d["m1"],
            d["m2"],
            d["m3"],
            d["R1"],
            d["R2"],
            d["k21"],
            d["k22"],
            d["ks1"],
            d["ks2"],
            d["tlag1"],
            d["tlag2"],
            d["alpha1"],
            d["alpha2"],
            int(perturber),
            int(gr_peri),
            int(tide1_peri),
            int(tide2_peri),
            int(spin1_peri),
            int(spin2_peri),
            int(evolve_spin2_axis),
            int(spin1_node),
            int(spin2_node),
            int(diss_tide1),
            int(diss_tide2),
            int(oct),
            int(mbraking1),
            int(mbraking2),
        )

        current_t, current_y, times, sol, a, e, peri, eps_gr, qflag = integration_loop(
            d,
            p,
            y_new,
            t_new,
            sol,
            times,
            a,
            e,
            peri,
            eps_gr,
            min_a,
            min_e,
            min_peri,
            max_eps_gr,
            True,
            False,
            meth="bdf",
            err=1e-9,
            resume=True,
        )

    # Classify outcomes
    if peri < min_peri:
        flag = 2
        sflag = "TD"
    elif (a < min_a) == True:
        flag = 1
        sflag = "HJ"

    elif min_a < a < 0.9 * d["a1"]:
        flag = 3
        sflag = "MG"

    elif current_t == t[-1]:
        flag = 0
        sflag = "NM"

    else:
        flag = 4
        sflag = "ERR"

    t = np.asarray(times)
    s = np.asarray(sol)

    print("Finished the integration. Quality flag: ", qflag)
    print("Outcome: ", sflag, flag)

    return t, s, flag, sflag, qflag
def set_inclinations(m1, m2, m3, a1, e1, a2, e2, Imut):

    ''' Determines the inclination of each orbit based on the specified mutual inclination so that the total angular momentum is along the z-axis'''

    def iterate(x1, x2, L1, L2, Itot):

        ''' Calculates the inclinations of the orbits so that the total angular momentum is along the z-axis '''

        def f(x, L1, L2, Itot):
            eq = L1 * np.sin(Itot - x) - L2 * np.sin(x)
            return eq

        imax = 10000
        Min = 1e-7

        a = x1
        b = x2

        i = 0
        while np.abs((b - a) / 2) > Min and i < imax:
            c = 0.5 * (a + b)
            f_a = f(a, L1, L2, Itot)
            f_b = f(b, L1, L2, Itot)
            f_c = f(c, L1, L2, Itot)

            if np.sign(f_a) == np.sign(f_c):
                a = c
            else:
                b = c

            i += 1
        if (i == imax):
            print("Max number of iterations reached")

        if ((np.abs(c / x2) < 1e-6) or (np.abs(c / x1) < 1e-6)):
            print("Did not find a root")
        return c

    mu1 = (m1 * m2) / (m1 + m2)
    mu2 = ((m1 + m2) * m3) / (m1 + m2 + m3)
    L1 = 2 * np.pi * mu1 * np.sqrt((m1 + m2) * a1 * np.sqrt(1 - e1 ** 2))
    L2 = 2 * np.pi * mu2 * np.sqrt((m1 + m2 + m3) * a2 * np.sqrt(1 - e2 ** 2))
    I2 = iterate(1e-9, np.pi, L1, L2, Imut)
    I1 = Imut - I2

    return I1, I2

def set_triple_properties(d, evolve_spin2_axis=True):

    """Takes the triple system parameters (masses, radii, orbital elements, etc) stored in a dictionary (d), calculates various quantities (angular momenta, etc) and returns a new dictionary with these quantities"""

    m1, m2, m3 = d["m1"], d["m2"], d["m3"]
    R1, R2, Pstar1, Pstar2 = d["R1"], d["R2"], d["Pstar1"], d["Pstar2"]
    k21, k22, ks1, ks2 = d["k21"], d["k22"], d["ks1"], d["ks2"]
    I1, peri1, node1 = d["I1"], d["peri1"], d["node1"]
    I2, peri2, node2 = d["I2"], d["peri2"], d["node2"]
    a1, e1, a2, e2 = d["a1"], d["e1"], d["a2"], d["e2"]

    m12 = m1 + m2
    m123 = m12 + m3
    mu_in = m1 * m2 / m12
    mu_out = m12 * m3 / m123

    # Tidal disruption radius
    Rtide = 2.7 * R2 * ((m1 / m2) ** (1.0 / 3.0))

    Omega1, Omega2 = 2 * np.pi / Pstar1, 2 * np.pi / Pstar2
    Omega1b = np.sqrt(G * m1 / (R1**3))
    Omega2b = np.sqrt(G * m2 / (R2**3))

    a2eff = a2 * np.sqrt(1 - e2**2)
    n1 = np.sqrt(G * m12 / (a1**3))
    n2 = np.sqrt(G * m123 / (a2**3))
    tk = (m12 / m3) * ((a2eff / a1) ** 3) * (1.0 / n1)

    L1 = mu_in * np.sqrt(G * m12 * a1 * (1 - e1**2))
    L2 = mu_out * np.sqrt(G * m123 * a2 * (1 - e2**2))
    S1, S2 = ks1 * m1 * R1 * R1 * Omega1, ks2 * m2 * R2 * R2 * Omega2

    L1x, L1y, L1z = (
        np.sin(I1) * np.sin(node1) * L1,
        -np.sin(I1) * np.cos(node1) * L1,
        np.cos(I1) * L1,
    )
    L2x, L2y, L2z = (
        np.sin(I2) * np.sin(node2) * L2,
        -np.sin(I2) * np.cos(node2) * L2,
        np.cos(I2) * L2,
    )

    e1x = (
        np.cos(peri1) * np.cos(node1) - np.sin(peri1) * np.cos(I1) * np.sin(node1)
    ) * e1
    e1y = (
        np.cos(peri1) * np.sin(node1) + np.sin(peri1) * np.cos(I1) * np.cos(node1)
    ) * e1
    e1z = np.sin(peri1) * np.sin(I1) * e1

    e2x = (
        np.cos(peri2) * np.cos(node2) - np.sin(peri2) * np.cos(I2) * np.sin(node2)
    ) * e2
    e2y = (
        np.cos(peri2) * np.sin(node2) + np.sin(peri2) * np.cos(I2) * np.cos(node2)
    ) * e2
    e2z = np.sin(peri2) * np.sin(I2) * e2

    # Place the spins initially parallel to the orbit
    S1x, S1y, S1z = (S1 / L1) * L1x, (S1 / L1) * L1y, (S1 / L1) * L1z
    S2x, S2y, S2z = (S2 / L1) * L1x, (S2 / L1) * L1y, (S2 / L1) * L1z

    # Calculate even more parameters of interest
    d_extra = calc_features(a1, Omega1, Omega2, e2, d)

    dnew = dict(d)
    dnew.update(
        {
            "L1x": L1x,
            "L1y": L1y,
            "L1z": L1z,
            "e1x": e1x,
            "e1y": e1y,
            "e1z": e1z,
            "L2x": L2x,
            "L2y": L2y,
            "L2z": L2z,
            "e2x": e2x,
            "e2y": e2y,
            "e2z": e2z,
            "S1x": S1x,
            "S1y": S1y,
            "S1z": S1z,
            "S2x": S2x,
            "S2y": S2y,
            "S2z": S2z,
            "L1": L1,
            "L2": L2,
            "S1": S1,
            "S2": S2,
            "Omega1": Omega1,
            "Omega2": Omega2,
            "Omega1b": Omega1b,
            "Omega2b": Omega2b,
            "m12": m12,
            "m123": m123,
            "mu_in": mu_in,
            "mu_out": mu_out,
            "a2eff": a2eff,
            "tk": tk,
            "n1": n1,
            "n2": n2,
            "Rtide": Rtide,
            "eps_gr":d_extra["eps_gr"],
            "eps_tide1": d_extra["eps_tide1"],
            "eps_tide2": d_extra["eps_tide2"],
            "eps_rot1": d_extra["eps_rot1"],
            "eps_rot2": d_extra["eps_rot2"],
            "eps_oct": d_extra["eps_oct"],
            "A1": d_extra["A1"],
            "A2": d_extra["A2"],
        }
    )

    return dnew


def process_output(t, sol, p, evolve_spin2_axis=True):

    """Takes the ODE solution array at times t, and parameters p, and calculates various quantities of interest. """

    L1vec = sol[:, 0:3]
    e1vec = sol[:, 3:6]
    L2vec = sol[:, 6:9]
    e2vec = sol[:, 9:12]
    S1vec = sol[:, 12:15]

    L1 = np.sqrt(L1vec[:, 0] ** 2 + L1vec[:, 1] ** 2 + L1vec[:, 2] ** 2)
    L2 = np.sqrt(L2vec[:, 0] ** 2 + L2vec[:, 1] ** 2 + L2vec[:, 2] ** 2)
    e1 = np.sqrt(e1vec[:, 0] ** 2 + e1vec[:, 1] ** 2 + e1vec[:, 2] ** 2)
    e2 = np.sqrt(e2vec[:, 0] ** 2 + e2vec[:, 1] ** 2 + e2vec[:, 2] ** 2)
    S1 = np.sqrt(S1vec[:, 0] ** 2 + S1vec[:, 1] ** 2 + S1vec[:, 2] ** 2)

    L_orb_x = L1vec[:, 0] + L2vec[:, 0]
    L_orb_y = L1vec[:, 1] + L2vec[:, 1]
    L_orb_z = L1vec[:, 2] + L2vec[:, 2]
    L_orb = np.sqrt(
        (L1vec[:, 0] + L2vec[:, 0]) ** 2
        + (L1vec[:, 1] + L2vec[:, 1]) ** 2
        + (L1vec[:, 2] + L2vec[:, 2]) ** 2
    )

    if evolve_spin2_axis == True:
        S2vec = sol[:, 15:18]
        S2 = np.sqrt(S2vec[:, 0] ** 2 + S2vec[:, 1] ** 2 + S2vec[:, 2] ** 2)

        L_spin_orb = np.sqrt(
            (S1vec[:, 0] + S2vec[:, 0] + L1vec[:, 0] + L2vec[:, 0]) ** 2
            + (S1vec[:, 1] + S2vec[:, 1] + L1vec[:, 1] + L2vec[:, 1]) ** 2
            + (S1vec[:, 2] + S2vec[:, 2] + L1vec[:, 2] + L2vec[:, 2]) ** 2
        )

    else:
        S2vec = np.nan * S1vec
        S2 = sol[:, 15]

        L_spin_orb = np.sqrt(
            (S1vec[:, 0] + L1vec[:, 0] + L2vec[:, 0]) ** 2
            + (S1vec[:, 1] + L1vec[:, 1] + L2vec[:, 1]) ** 2
            + (S1vec[:, 2] + L1vec[:, 2] + L2vec[:, 2]) ** 2
        )

    delta_L_orb = np.abs((L_orb - L_orb[0]) / L_orb[0])
    delta_L_spin_orb = np.abs((L_spin_orb - L_spin_orb[0]) / L_spin_orb[0])

    cos_theta1 = (
        S1vec[:, 0] * L1vec[:, 0]
        + S1vec[:, 1] * L1vec[:, 1]
        + S1vec[:, 2] * L1vec[:, 2]
    ) / (S1 * L1)
    cos_theta2 = (
        S2vec[:, 0] * L1vec[:, 0]
        + S2vec[:, 1] * L1vec[:, 1]
        + S2vec[:, 2] * L1vec[:, 2]
    ) / (S2 * L1)
    cos_Imut = (
        L1vec[:, 0] * L2vec[:, 0]
        + L1vec[:, 1] * L2vec[:, 1]
        + L1vec[:, 2] * L2vec[:, 2]
    ) / (L1 * L2)

    # Argument of pericenter and longitdue of ascending node
    l1hat = L1vec / L1[:, None]
    l2hat = L2vec / L2[:, None]
    node1 = np.arctan2(-l1hat[:, 0], l1hat[:, 1])
    node2 = np.arctan2(-l2hat[:, 0], l2hat[:, 1])

    I1 = np.arctan2(l1hat[:, 0], l1hat[:, 2] * np.sin(node1))
    I2 = np.arctan2(l2hat[:, 0], l2hat[:, 2] * np.sin(node2))

    e1hat = e1vec / e1[:, None]
    e2hat = e2vec / e2[:, None]
    z = np.zeros([len(e1), 3])
    z[:, 2] = 1.0
    x1hat = np.cross(z, l1hat / np.sin(I1[:, None]))
    cosw1 = (
        e1hat[:, 0] * x1hat[:, 0]
        + e1hat[:, 1] * x1hat[:, 1]
        + e1hat[:, 2] * x1hat[:, 2]
    )
    sinw1 = e1hat[:, 2] / np.sin(I1)
    w1 = np.arctan2(sinw1, cosw1)

    x2hat = np.cross(z, l2hat / np.sin(I2[:, None]))
    cosw2 = (
        e2hat[:, 0] * x2hat[:, 0]
        + e2hat[:, 1] * x2hat[:, 1]
        + e2hat[:, 2] * x2hat[:, 2]
    )
    sinw2 = e2hat[:, 2] / np.sin(I2)
    w2 = np.arctan2(sinw2, cosw2)

    theta1 = np.arccos(cos_theta1)
    if evolve_spin2_axis == True:
        theta2 = np.arccos(cos_theta2)
    else:
        theta2 = np.nan*theta1
    Imut = np.arccos(cos_Imut)
    
    a1 = ((L1 / p["mu_in"]) ** 2) * (1.0 / (G * p["m12"])) * (1.0 / (1 - e1**2))
    Omega1 = S1 / p["Imom1"]
    Omega2 = S2 / p["Imom2"]

    d = {
        "simnum":p["simnum"],
        "t": t,
        "e1": e1,
        "L1": L1,
        "e2": e2,
        "L2": L2,
        "S1": S1,
        "S2": S2,
        "L_orb": L_orb,
        "L_spin_orb": L_spin_orb,
        "theta1": theta1,
        "theta2": theta2,
        "Imut": Imut,
        "L1x": L1vec[:, 0],
        "L1y": L1vec[:, 1],
        "L1z": L1vec[:, 2],
        "e1x": e1vec[:, 0],
        "e1y": e1vec[:, 1],
        "e1z": e1vec[:, 2],
        "L2x": L2vec[:, 0],
        "L2y": L2vec[:, 1],
        "L2z": L2vec[:, 2],
        "e2x": e2vec[:, 0],
        "e2y": e2vec[:, 1],
        "e2z": e2vec[:, 2],
        "S1x": S1vec[:, 0],
        "S1y": S1vec[:, 1],
        "S1z": S1vec[:, 2],
        "peri1": w1,
        "peri2": w2,
        "node1": node1,
        "node2": node2,
        "I1": I1,
        "I2": I2,
        "L_orb_x": L_orb_x,
        "L_orb_y": L_orb_y,
        "L_orb_z": L_orb_z,
        "delta_L_spin_orb": delta_L_spin_orb,
        "delta_L_orb": delta_L_orb,
        "a1": a1,
        "Omega1": Omega1,
        "Omega2": Omega2,
        "Pspin1": 2 * np.pi / Omega1,
        "Pspin2": 2 * np.pi / Omega2,
    }

    df = pd.DataFrame(d)

    return df


def calc_a_e(sol, m1, m2):

    """Takes the ODE solution and calculates the semi-major axis and eccentricity from the anglular momentum and eccentricity vectors"""

    L1vec = sol[0:3]
    e1vec = sol[3:6]

    e1 = np.sqrt(e1vec[0] ** 2 + e1vec[1] ** 2 + e1vec[2] ** 2)
    L1 = np.sqrt(L1vec[0] ** 2 + L1vec[1] ** 2 + L1vec[2] ** 2)

    m12 = m1 + m2
    mu = m1 * m2 / m12

    a1 = ((L1 / mu) ** 2) * (1.0 / (G * m12)) * (1.0 / (1 - e1**2))

    return a1, e1, a1 * (1 - e1)


def calc_features(a1, Omega1, Omega2, e2, parameters):

    """Calculates various parameters of interest: the strengths of the short-range forces compared to the perturber, as defined in Liu, Munoz, & Lai 2015.
    Also calculates strength of the octupole term, and A parameters describing the spin-orbit coupling of the primary and secondary (Anderson, Lai, & Storch 2017).
    Input is the potentially time-dependent semi-major axis (a1), spin rates (Omega1,Omega2), and outer binary eccentricity (e2), along with a dictionary of parameters that are constant in time (parameters)."""

    m1, m2, m3 = parameters["m1"], parameters["m2"], parameters["m3"]
    m12 = m1 + m2

    k21, k22 = parameters["k21"], parameters["k22"]
    ks1, ks2 = parameters["ks1"], parameters["ks2"]
    kq1, kq2 = (2.0 / 3.0) * k21, (2.0 / 3.0) * k22

    R1, R2 = parameters["R1"], parameters["R2"]

    a2 = parameters["a2"]

    a2eff = a2 * np.sqrt(1 - e2**2)
    n1 = np.sqrt(G * m12 / (a1**3))
    Porb1 = 2*np.pi/n1
    tk = (m12 / m3) * ((a2eff / a1) ** 3) * (1.0 / n1)

    Omega1b, Omega2b = np.sqrt(G * m1 / (R1**3)), np.sqrt(G * m2 / (R2**3))
    Omega_hat1, Omega_hat2 = Omega1 / Omega1b, Omega2 / Omega2b

    # Apsidal precession rates, without e-dependence of the inner binary
    wdot_gr = 3 * m12 * n1 / ((clight**2) * a1)

    wdot_tide1 = 15 * n1 * (m2 / m1) * k21 * (R1 / a1) ** 5
    wdot_tide2 = 15 * n1 * (m1 / m2) * k22 * (R2 / a1) ** 5

    wdot_rot1 = 1.5 * kq1 * ((R1 / a1) ** 2) * (Omega_hat1**2) * n1
    wdot_rot2 = 1.5 * kq2 * ((R2 / a1) ** 2) * (Omega_hat2**2) * n1

    # Precession of the spin axes, without e-dependence
    Omega_s1 = 1.5 * (kq1 / ks1) * (m2 / m1) * ((R1 / a1) ** 3) * Omega1
    Omega_s2 = 1.5 * (kq2 / ks2) * (m1 / m2) * ((R2 / a1) ** 3) * Omega2

    # Octupole strength
    eps_oct = ((m1 - m2) / m12) * (a1 / a2) * e2 / (1 - e2**2)

    # Strength of short-range forces compared to perturber
    eps_gr = wdot_gr * tk
    eps_tide1 = wdot_tide1 * tk
    eps_tide2 = wdot_tide2 * tk
    eps_rot1 = wdot_rot1 * tk
    eps_rot2 = wdot_rot2 * tk

    # Adiabicity parameters
    A1, A2 = Omega_s1 * tk, Omega_s2 * tk

    d_features = {
        "simnum":parameters["simnum"],
        "eps_gr": eps_gr,
        "eps_tide1": eps_tide1,
        "eps_tide2": eps_tide2,
        "eps_rot1": eps_rot1,
        "eps_rot2": eps_rot2,
        "eps_oct": eps_oct,
        "A1": A1,
        "A2": A2,
        "tk": tk,
        "Porb1":Porb1
    }

    return d_features



