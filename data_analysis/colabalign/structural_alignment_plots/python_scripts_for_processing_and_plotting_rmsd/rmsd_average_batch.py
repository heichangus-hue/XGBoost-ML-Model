import numpy as np
import pandas as pd
from pathlib import Path

def calculate_batch_rmsd(results_root):
    results_path = Path(results_root)
    summary_data = []

    # Check if the path actually exists
    if not results_path.exists():
        print(f"Error: The path {results_path} does not exist.")
        return

    # Indent everything to stay inside the loop
    for protein_folder in results_path.iterdir():
        if not protein_folder.is_dir():
            continue
    
        pdb_code = protein_folder.name
        
        # Search for files
        ref_list = list(protein_folder.glob("*_reference_coords.csv"))
        trans_list = list(protein_folder.glob("*_cofactor_aligned_coords.csv"))

        # Skip this folder if files are missing, but don't stop the whole script
        if not ref_list or not trans_list:
            print(f"Skipping {pdb_code}: CSV files not found.")
            continue

        # Load the data
        data_file_reference = pd.read_csv(ref_list[0])
        data_file_transformed = pd.read_csv(trans_list[0])

        # Merge data on index to ensure alignment
        #merged_data = pd.merge(
            #data_file_reference,
            #data_file_transformed,
            #left_index=True,
            #right_index=True,
            #suffixes=('_ref', '_trans')
        #)

        merged_data = pd.merge(
                data_file_reference,
                data_file_transformed,
                on=['chain', 'res_id', 'atom'],
                suffixes=('_ref', '_trans')
                )

        if merged_data.empty:
            print(f"Skipping {pdb_code}: No matching atoms found in merge.")
            continue

        # Extract coordinates and calculate RMSD
        coords_reference = merged_data[['x_ref', 'y_ref', 'z_ref']].values
        coords_transformed = merged_data[['x_trans', 'y_trans', 'z_trans']].values

        diff = coords_reference - coords_transformed
        squared_distances = np.sum((diff)**2, axis=1) 
        mean_squared_distance = np.mean(squared_distances)
        rmsd = np.sqrt(mean_squared_distance)

        summary_data.append({
            "PDB_Code": pdb_code,
            "RMSD": round(rmsd, 4),
            "Atoms_Compared": len(merged_data)
        })

        print(f"Processed {pdb_code}: RMSD = {rmsd:.4f}") 

    # --- After the loop is finished, save the summary ---
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        output_file = results_path / "batch_rmsd_summary_new.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\n--- Summary saved to {output_file} ---")
    else:
        print("Error: No data was successfully processed.")

# IMPORTANT: Ensure the path points to the folder that contains the sub-folders
calculate_batch_rmsd("/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4")



