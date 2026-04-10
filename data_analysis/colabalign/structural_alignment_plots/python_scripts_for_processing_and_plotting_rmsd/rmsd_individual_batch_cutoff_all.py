import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def calculate_individual_rmsd(results_root, master_distance_csv):
    results_path = Path(results_root)
    summary_data = []

    # 1. Load the unified distance file
    if not Path(master_distance_csv).exists():
        print(f"Error: Distance file not found at {master_distance_csv}")
        return

    print(f"Loading master distances from: {master_distance_csv}")
    master_dist_df = pd.read_csv(master_distance_csv)
    
    # Normalize PDB IDs and clean strings
    master_dist_df['PDB_ID'] = master_dist_df['PDB_ID'].astype(str).str.lower()
    master_dist_df['res_id_str'] = master_dist_df['res_id'].astype(str).str.strip()
    master_dist_df['atom_clean'] = master_dist_df['atom'].astype(str).str.strip().str.upper()

    for protein_folder in results_path.iterdir():
        if not protein_folder.is_dir():
            continue

        pdb_code = protein_folder.name.lower()

        # 2. Filter distance data for this specific PDB
        this_pdb_dist = master_dist_df[master_dist_df['PDB_ID'] == pdb_code].copy()
        
        if this_pdb_dist.empty:
            continue

        # Find the coordinate CSV files
        ref_list = list(protein_folder.glob("*_reference_coords.csv"))
        trans_list = list(protein_folder.glob("*_cofactor_aligned_coords.csv"))

        if not ref_list or not trans_list:
            continue

        # Load Coordinate Data
        df_ref = pd.read_csv(ref_list[0])
        df_trans = pd.read_csv(trans_list[0])

        # 3. Merge Coordinate files (Reference + Transformed)
        merged_data = pd.merge(
            df_ref, df_trans, 
            on=['chain', 'res_id', 'atom'], 
            suffixes=('_ref', '_trans')
        )


        # Print lengths before final merge --> Row count check for debugging
        print(f"\nRow Check for {pdb_code}:")
        print(f"  Rows in RMSD coordinates: {len(merged_data)}")
        print(f"  Rows in Distance CSV:     {len(this_pdb_dist)}")

        # 4. Preparation for BIOLOGICAL MERGE (Residue ID + Atom Name)
        merged_data['res_id_str'] = merged_data['res_id'].astype(str).str.strip()
        merged_data['atom_clean'] = merged_data['atom'].astype(str).str.strip().str.upper()

        # 5. Perform the Merge on Biological Identifiers
        # This resolves the 9-atom shift by ignoring index numbers
        merged_data = pd.merge(
            merged_data,
            this_pdb_dist[['res_id_str', 'atom_clean', 'distance']],
            on=['res_id_str', 'atom_clean'],
            how='left'
        )
        
        # Check how many distance values successfully mapped
        matched_count = merged_data['distance'].notna().sum()
        print(f"  Atoms successfully matched: {matched_count}")

        # 6. Calculate RMSD
        coords_ref = merged_data[['x_ref', 'y_ref', 'z_ref']].values
        coords_trans = merged_data[['x_trans', 'y_trans', 'z_trans']].values
        per_index_rmsd = np.sqrt(np.sum((coords_ref - coords_trans)**2, axis=1))
        merged_data['per_index_rmsd'] = per_index_rmsd

        # 7. Create Superimposed Overlay Logic
        #cutoff_val = [5.0, 10.0, 15.0, 20.0]  # Define multiple cutoff values for comparison
        #colors = ['darkorange', 'royalblue', 'seagreen', 'crimson']  # Colours for each cutoff
        
        # Format: (label, lower_bound, upper_bound, color)
        shells = [
            (f'0-5 \u00c5',   0,  5,  'darkorange'),
            (f'5-10 \u00c5',  5,  10, 'royalblue'),
            (f'10-15 \u00c5', 10, 15, 'forestgreen'),
            (f'15-20 \u00c5', 15, 20, 'crimson')
        ]


        #merged_data['filtered_rmsd'] = np.where(
            #merged_data['distance'] <= cutoff_val, 
            #merged_data['per_index_rmsd'], 
            #np.nan
        #)

        # 8. Plotting
        plt.figure(figsize=(16, 8))
        
        plt.plot(merged_data.index, merged_data['per_index_rmsd'], 
                 label='Full Protein RMSD', color='lightgray', alpha=0.6, linewidth=1)
        
        # Loop through Loop through cutoffs (plotting larger ones first or using zorder, so smaller distances stay visible on top)

        for label, lower, upper, color in shells:
            merged_data[f'filtered_rmsd_{label}'] = np.where(
                (merged_data['distance'] > lower) & (merged_data['distance'] <= upper), 
                merged_data['per_index_rmsd'], 
                np.nan
            )

            plt.plot(merged_data.index, merged_data[f'filtered_rmsd_{label}'], 
                    label=f'{lower} < $d$ ≤ {upper} \u00c5 Near Cofactor', color=color, linewidth=2.5)

        #plt.plot(merged_data.index, merged_data['filtered_rmsd'], 
                 #label=f'Near Cofactor (< {cutoff_val} \u00c5)', color='darkorange', linewidth=2.5)

        
        plt.title(f'Structural Deviation Overlay: {pdb_code.upper()}', size=14)
        plt.xlabel('Index', size=12)
        plt.ylabel('RMSD / \u00c5', size=12)
        #plt.legend(loc='upper right')
        plt.legend(bbox_to_anchor=(1.02, 0.6), loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        save_path = protein_folder / f"{pdb_code}_rmsd_distance_overlay_all.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"  Overlay plot saved to: {save_path}")

# EXECUTION
RESULTS_DIR = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4"
DISTANCE_CSV = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/CIF/final_atomic_distances.csv"

calculate_individual_rmsd(RESULTS_DIR, DISTANCE_CSV)
