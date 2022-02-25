# Overview
The code in this repository models the dynamical evolution of astrophysical triple systems, using the secular (orbit-averaged) approximation. This significantly speeds up the simulation run time compared to full N-body simulations. As a result of the secular approximation, the tertiary body must be sufficiently distant from the other bodies for the simulation output to be valid. The time evolution of the system is described by a system of coupled ordinary differential equations (see [Anderson, Lai, & Storch 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.3671A/abstract)). 

Current physical effects implemented in the code include:
- Gravitational torques (up to octupole-order in the semi-major axis ratio)
- Apsidal precession due to general relativity, tides, and oblateness
- Tidal dissipation
- Spin-orbit coupling
- Magnetic braking due to stellar winds

These effects can be turned on or off by the user as desired. 

Applications of this code include triple star systems, planets in binary star systems, and multi-planet systems (as in [Anderson, Lai, & Storch 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.3671A/abstract), [Anderson, Lai, & Storch 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.3066A/abstract) and [Anderson & Lai 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3692A/abstract)). 

# Running a single simulation
A Jupyter notebook tutorial is provided (see `notebooks/migrating_planet_example.ipynb`). In brief, the user must specify the properties of each body (e.g. mass and radius), along with the initial conditions for the ODE solver. Code for processing the user-defined input parameters, solving the ODEs, and processing the ODE solution is found in `src/integrate_odes.py`. Built into the code are several stopping conditions for the ODE solver, with changeable parameters that the user may tune to their application. The ODES are solved using the BDF method from the Scipy [`ode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html) solver.

# Running many simulations in parallel using Monte-Carlo sampling
Python/bash scripts for running many simulations in parallel are provided in the `run_in_batch` directory. The file `create_ic.py` shows an example of sampling various system parameters and initial conditions from distributions, which are then written to a series of files (one file per simulation). The script `run_from_file.py` reads an input file, performs the simulation by passing the input parameters to the code in `src/integrate_odes.py`, and saves the desired output to a file. The `run_parallel.bash` script generates the initial conditions by running the `create_ic.py` script, and then runs the simulations in parallel using GNU Parallel and passing each set of initial conditions to the `run_from_file.py` script.

# Libraries

The Python code requires NumPy, SciPy, Pandas, Astropy, and Matplotlib to be installed. In order to run the `run_parallel.bash` script in the `run_in_batch` directory, [GNU Parallel](https://www.gnu.org/software/parallel/) must be installed.


