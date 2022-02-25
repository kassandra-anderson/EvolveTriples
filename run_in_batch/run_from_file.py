import numpy as np
import pandas as pd
import sys,os,time
sys.path.insert(1, '../src/')
import integrate_odes as io

print("---------------------------------------------------------------")

#Start time (to monitor code performance)
t_code_start = time.time()

# Assign input and output files
infile = str(sys.argv[1])
outfile = str(sys.argv[2])

# Read input parameters from file
df_input = pd.read_csv(infile,delimiter=',')
parameters = df_input.iloc[0]

# Update the parameter dictionary with additional quantities
parameters = io.set_triple_properties(parameters)

# Stopping conditions to pass to io.integrate_triple
min_peri = parameters["Rtide"]
min_a, min_e = 0.1, 0.5

# Integration timespan (yr).
# tend can be negative to integrate backwards in time.
# Can set tend = x * parameters["tk"] to integrate for x secular timescales
tend = 1e7

# Specify density of the ODE solution output
# Npoints = 5000000 # for popsynth
Npoints = 10000

t = np.linspace(0, tend, Npoints)

# Integrate the ODEs
t, sol, flag, sflag, qflag = io.integrate_triple(
    parameters,
    t,
    evolve_spin2_axis=False,
    spin2_node=False,
    diss_tide2=True,
    mbraking1=True,
    min_peri=min_peri,
    min_a=min_a,
    min_e=min_e,
)

t_code_end = time.time()
# Code runtime in hours
runtime =  (t_code_end - t_code_start)/3600

# Process the output of the ODEs
df = io.process_output(t, sol, parameters, evolve_spin2_axis=False)

# Extract the properties at the end of the integration
df_end = df.tail(1)

# Add information about ODE solution quality and outcome, runtime, and random number seed
df_end = df_end.assign(flag=flag,sflag=sflag,qflag=qflag,runtime=runtime,seed=parameters["seed"])

# Calculate various parameters of interest, store in a dictionary
d_features = io.calc_features(
    df_end["a1"].values, df_end["Omega1"].values, df_end["Omega2"].values, df_end["e2"].values, parameters
)

# Add these additional parameters to existing dataframe
df_feat = pd.DataFrame(d_features,index=[parameters["simnum"]])

# Convert input parameters from dictionary to dataframe
df_in = pd.DataFrame(parameters,index=[parameters["simnum"]])

# Rename the input paramater columns to avoid confusion with final parameter columns
df_in = df_in.rename(columns={col: col+'_input' for col in df_in.columns if col not in ["simnum"]})

# Create a dataframe storing the initial conditions and ODE solution at the final timestep.
df_join = pd.merge(pd.merge(df_in,df_end,on="simnum"),df_feat,on="simnum")

# Create or append to output file
if not os.path.exists(outfile):
    df_join.to_csv(outfile, sep=',',mode='w',index=False,header=True)
    print("Creating output file")
else:
    df_join.to_csv(outfile, sep=',',mode='a',index=False,header=False)
    print("Appending to output file")
print("---------------------------------------------------------------")