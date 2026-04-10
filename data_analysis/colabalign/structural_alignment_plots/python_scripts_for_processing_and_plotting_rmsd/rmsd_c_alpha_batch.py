import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def calculate_individual_rmsd(results_root):
    results_path = Path(results_root)
    summary_data = []

    if not results_path.exists():
        print(f"Error: The path {results_path} does not exist.")
        return

    for protein_folder in results_path.iterdir():
        if not protein_folder.is_dir():
            continue

        pdb_code = protein_folder.name

        # Search for files
        ref_list = list(protein_folder.glob("*_reference_coords.csv"))
        trans_list = list(protein_folder.glob("*_cofactor_aligned_coords.csv"))

        if not ref_list or not trans_list:
            print(f"Skipping {pdb_code}: CSV files not found.")
            continue

        # Load the data
        data_file_reference = pd.read_csv(ref_list[0])
        data_file_transformed = pd.read_csv(trans_list[0])

        df_reference_ca = data_file_reference[data_file_reference['atom'] == 'CA'].copy()
        df_transformed_ca = data_file_transformed[data_file_transformed['atom'] == 'CA'].copy()

        print(f"\nProcessing {pdb_code}:")
        print(f"  Number of rows in reference: {len(df_reference_ca)}")
        print(f"  Number of rows in transformed: {len(df_transformed_ca)}")

        merged_ca = pd.merge(
                df_reference_ca,
                df_transformed_ca,
                on=['chain', 'res_id', 'atom'],
                suffixes=('_ref', '_trans')
                )

        if merged_ca.empty:
            print(f"Skipping {pdb_code}: No matching atoms found in merge.")
            continue

        # Extract coordinates and calculate RMSD
        coords_reference = merged_ca[['x_ref', 'y_ref', 'z_ref']].values
        coords_transformed = merged_ca[['x_trans', 'y_trans', 'z_trans']].values

        diff = coords_reference - coords_transformed
        squared_distances = np.sum((diff)**2, axis=1) # axis = 1 looks at each column
        per_ca_rmsd = np.sqrt(squared_distances)
        #rmsd = np.sqrt(mean_squared_distance)
        
        merged_ca['per_ca_rmsd'] = per_ca_rmsd

        #overall_max = per_ca_rmsd.max()
        #summary_data.append({
            #'pdb_code': pdb_code,
            #'max_RMSD': per_ca_rmsd.max()
        #})
        #print(f"  Max RMSD: {overall_max:.4f} \u00c5")

        plt.figure(figsize=(10, 8))
        
        plt.plot(merged_ca.index, merged_ca['per_ca_rmsd'], label='RMSD per CA Atom')
        plt.xlabel(r'C$_\alpha$ Atoms', size=14)
        plt.ylabel('RMSD / $\mathrm{\AA}$', size=14)
        plt.title(rf'Structural Deviation by C$_\alpha$ Atoms for {pdb_code.upper()}', size=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        plt.tight_layout()
        save_path = protein_folder / f"{pdb_code}_rmsd_per_ca_atom.png"
        plt.savefig(save_path, dpi = 500)
        plt.close()
        print(f"  > Plot saved to: {save_path}")

    #if summary_data:
        #summary_df = pd.DataFrame(summary_data)
        #summary_df = summary_df.sort_values(by='max_RMSD', ascending=False) #Sort by max RMSD
        #output_csv = results_path / "batch_max_rmsd_summary.csv"
        #summary_df.to_csv(output_csv, index=False)
        #print(f"\nRMSD summary saved to {output_csv}")

calculate_individual_rmsd("/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4")

        

