import pandas as pd
from pathlib import Path

# Path to your main results directory
results_dir = Path("/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4")

# Loop through every subfolder
for ref_file in results_dir.glob("**/*_reference_coords.csv"):
    print(f"Updating: {ref_file}")
    
    # Read the CSV
    df = pd.read_csv(ref_file)
    
    # Rename the specific columns
    df = df.rename(columns={
        'x': 'x_ref',
        'y': 'y_ref',
        'z': 'z_ref'
    })
    
    # Save it back to the same location
    df.to_csv(ref_file, index=False)

print("All reference coordinate headers updated.")


