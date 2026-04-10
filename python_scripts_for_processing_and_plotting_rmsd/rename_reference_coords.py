import pandas as pd
from pathlib import Path

def standardize_reference_indices(results_root):
    root_path = Path(results_root)
    
    # Use glob to find all reference files in all subdirectories
    ref_files = list(root_path.rglob("*reference_coords.csv"))
    
    print(f"Found {len(ref_files)} reference files to process...")

    for file_path in ref_files:
        # 1. Load the data
        df = pd.read_csv(file_path)

        # 2. Check if 'index' already exists as a column and remove it 
        # to prevent "index_x" or "index_y" duplicates later
        if 'index' in df.columns:
            df = df.drop(columns=['index'])

        # 3. Reset the index. 
        # drop=False moves the current index into a column.
        # This creates a fresh 0 to N sequence.
        df = df.reset_index() 
        
        # 4. Optional: Rename 'index' to something specific if preferred
        # df = df.rename(columns={'index': 'atom_serial'})

        # 5. Overwrite the file with the standardized version
        df.to_csv(file_path, index=False)
        print(f"  Standardized: {file_path.parent.name}")

# Run this once before running your RMSD script
standardize_reference_indices("/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4")

