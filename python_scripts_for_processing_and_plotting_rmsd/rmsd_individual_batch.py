import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

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
            print(f"SkippP1+r4632=1B5B32347E\ing {pdb_code}: CSV files not found.")
            continue

        # Load the data
        data_file_reference = pd.read_csv(ref_list[0])
        data_file_transformed = pd.read_csv(trans_list[0])

        print(f"\nProcessing {pdb_code}:")
        print(f"  Reference rows:   {len(data_file_reference)}")
        print(f"  Transformed rows: {len(data_file_transformed)}")

        merged_data = pd.merge(
                data_file_reference,
                data_file_transformed,
                on=['chain', 'res_id', 'atom'],
                suffixes=('_ref', '_trans')
                )

        if merged_data.empty:
            print(f"Skipping {pdb_code}: No matching atoms found in merge.")
            continue

        # Extract coordinates and calculate P0+r2531\P0+r2638\P1+r6B62=7F\P0+r6B49\P1+r6B44=1B5B337E\P1+r6B68=1B4F48\P1+r4037=1B4F46\P1+r6B50=1B5B357E\P1+r6B4E=1B5B367E\RMSD
        coords_reference = merged_data[['x_ref', 'y_ref', 'z_ref']].values
        coords_transformed = merged_data[['x_trans', 'y_trans', 'z_trans']].values

        diff = coords_reference - coords_transformed
        squared_distances = np.sum((diff)**2, axis=1) # axis = 1 looks at each column
        per_index_rmsd = np.sqrt(squared_distances)
        #rmsd = np.sqrt(mean_squared_distance)
        
        merged_data['per_index_rmsd'] = per_index_rmsd

        overall_max = per_index_rmsd.max()
        summary_data.append({
            'pdb_code': pdb_code,
            'max_RMSD': overall_max
        })
        print(f"  Max RMSD: {overall_max:.4f} \u00c5")

        plt.figure(figsize=(10, 8))
        
        plt.plot(merged_data.index, merged_data['per_index_rmsd'], label='RMSD per Index')
        plt.xlabel('Index', size=14)
        plt.ylabel('RMSD / $\mathrm{\AA}$', size=14)
        plt.title(rf'Structural Deviation by Index for {pdb_code.upper()}', size=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        plt.tight_layout()
        save_path = protein_folder / f"{pdb_code}_rmsd_per_index.png"
        plt.savefig(save_path, dpi = 500)
        plt.close()
        print(f"  > Plot saved to: {save_path}")
        
        merged_data = merged_data.reset_index(names='index')

        print(merged_data[['index', 'per_index_rmsd']])
        output_csv_2 = protein_folder / f"{pdb_code}_rmsd.csv"
        merged_data[['index', 'res_id', 'atom', 'per_index_rmsd']].to_csv(output_csv_2, index=False)

        print(f"  > RMSD data saved to: {output_csv_2}")
        
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by='max_RMSD', ascending=False) #Sort by max RMSD
        output_csv = results_path / "batch_max_rmsd_summary.csv"
        summary_df.to_csv(output_csv, index=False)
        print(f"\nRMSD summary saved to {output_csv}")

calculate_individual_rmsd("/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4")
        

