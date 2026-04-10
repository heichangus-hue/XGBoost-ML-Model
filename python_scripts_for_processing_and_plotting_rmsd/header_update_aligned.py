import pandas as pd
from pathlib import Path

# Path to your main results directory
results_dir = Path("/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4")

# Loop through every subfolder
for aligned_file in results_dir.glob("**/*_aligned_coords.csv"):
    print(f"Updating: {aligned_file}")
    
    # Read the CSV
    df = pd.read_csv(aligned_file)
    
    # Rename the specific columns
    df = df.rename(columns={
        'x': 'x_trans',
        'y': 'y_trans',
        'z': 'z_trans'
    })
    
    # Save it back to the same location
    df.to_csv(aligned_file, index=False)

print("All reference coordinate headers updated.")


