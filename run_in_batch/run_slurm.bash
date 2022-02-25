#!/bin/bash
#SBATCH --ntasks=4
#SBATCH -N 1
#SBATCH --time=0:2:00
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=kra2@princeton.edu 

module load anaconda3/2021.11
source activate ds


# Notes: need to set parameters correctly in run_from_file.py

# Set resume = 1 if resuming
resume=0

# Seed for the random number generator
seed=1

# Number of systems to sample. IMPORTANT: need to set number of systems separately in parallel command below {1..Nrun}, because of BASH limitations.
Nrun=4

outpath="/scratch/gpfs/kra2/simulations/seed""$seed"/""
icpath=$outpath

outfile="$outpath"batch_seed"$seed".csv""
logfile="$icpath"logfile.log""

run_int() {

	i="$1"
	icpath="$2"
	outfile="$3"
	infile="$icpath"ic"$i".csv""

	# Run simulation
	python run_from_file.py "$infile" "$outfile"

	i=$(( $i + 1 ))
}

export -f run_int

if [ "$resume" -eq 0 ]; then
    echo "Generating initial conditions"
    rm -rf "$icpath" ; mkdir -p "$icpath"
    python create_ic.py "$icpath" "$Nrun" "$seed"
fi

parallel --delay .2 -j "$SLURM_NTASKS" --joblog "$logfile" --resume run_int ::: {1..4} ::: "$icpath" ::: "$outfile"
echo Finished




