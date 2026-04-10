#!/bin/bash
#SBATCH --job-name=protein_ML    # Job name
#SBATCH --partition=serial       # Use the serial partition (for CPU jobs)
#SBATCH --ntasks=1               # Run on a single CPU
#SBATCH --mem=4G                 # Request 4GB of RAM
#SBATCH --time=08:00:00          # Limit to 10 minutes (it will likely take seconds)
#SBATCH --output=ml_results.out  # This is where your print statements will go

# Load the environment you were using (if you use conda/miniforge)
# source ~/miniforge3/bin/activate base

# Run my script
python testing_XG_boost.py
