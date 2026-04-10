import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import linregress

# 1. Configuration & Paths
pdb_codes = [
    "1a8z", "1e1l", "1owp", "2aqj", "2weg", "3nlc", "3t87", "4nvn", "6ugr", "7xj4",
    "1bxa", "1e1n", "1pha", "2aw1", "2xv2", "3nt0", "3wec", "5grt", "7g3b", "7y2e",
    "1cri", "1ft7", "1pxc", "2cak", "2yym", "3orj", "4bte", "5msz", "7hux", "8a1h",
    "1d3s", "1h3n", "1reo", "2cal", "3axb", "3p6n", "4cxo", "5nqm", "7og7", "8ewq",
    "1dob", "1hfc", "1sog", "2e0p", "3dso", "3p6u", "4d3t", "5ns5", "7or2", "8jdy",
    "1ds4", "1jzg", "1vxa", "2ft7", "3fg2", "3pb9", "4dpc", "5ssz", "7pq1", "8z3g",
    "1dve", "1of0", "1yso", "2h2i", "3fkg", "3qm8", "4mai", "6km5", "7rkr", "8z44",
    "1dz0", "1ogi", "1yzp", "2j18", "3f7l", "3qqx", "4nva", "6rtm", "7uhi", "8z45"
]

# Note: Check that these base paths are correct for your environment
path_to_rmsd_template = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/ColabAlign/FINAL_4/{pdb_code}/{pdb_code}_rmsd.csv"
path_to_plddt_template = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all/{pdb_code}/seed-1_sample-0/confidences.json"

results_summary = []

print("Starting batch processing...")

# 2. The Main Loop
for pdb in pdb_codes:
    rmsd_file = path_to_rmsd_template.format(pdb_code=pdb)
    plddt_file = path_to_plddt_template.format(pdb_code=pdb)
    
    # Check if both files exist
    if not os.path.exists(rmsd_file) or not os.path.exists(plddt_file):
        print(f"Skipping {pdb}: Files missing.")
        continue

    try:
        # Load RMSD Data
        rmsd_df = pd.read_csv(rmsd_file)
        # Using your specific column name
        rmsd_vals = rmsd_df['per_index_rmsd'].values
        
        # Load pLDDT Data from JSON
        with open(plddt_file, 'r') as f:
            conf_data = json.load(f)
            # Try 'plddt' first, then 'atom_plddts'
            plddt_vals = np.array(conf_data.get('plddt', conf_data.get('atom_plddts')))

        # Handle potential length mismatches (direct append/trimming)
        min_len = min(len(rmsd_vals), len(plddt_vals))
        x = rmsd_vals[:min_len]
        y = plddt_vals[:min_len]

        # Clean NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean, y_clean = x[mask], y[mask]

        if len(x_clean) < 2:
            print(f"Skipping {pdb}: Not enough data points after cleaning.")
            continue

        # 3. Math - Linear Regression
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        r_squared = r_value**2
        results_summary.append((pdb, r_squared))

        # 4. Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(x_clean, y_clean, alpha=0.5, s=15, color='tab:blue', label='Data Points')
        
        # Regression Line Plot
        line = intercept + slope * x_clean
        plt.plot(x_clean, line, 'r-', linewidth=2, label=f'Fit: $R^2 = {r_squared:.4f}$')

        # Formatting
        plt.xlabel(r'RMSD / $\AA$')
        plt.ylabel('pLDDT')
        plt.title(f'pLDDT vs RMSD for {pdb.upper()}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # 5. Saving to the specific directory
        output_dir = f"/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all/{pdb}"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        save_path = os.path.join(output_dir, f"{pdb}_plddt_vs_rmsd.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Close plot to prevent memory leaks
        plt.close()
        print(f"Successfully processed {pdb} (R^2 = {r_squared:.3f})")

    except Exception as e:
        print(f"Error processing {pdb}: {str(e)}")

# 6. Final Summary
print("\n" + "="*30)
print(f"Processing Complete.")
print(f"Total Successful Plots: {len(results_summary)}")
if results_summary:
    avg_r2 = np.mean([val[1] for val in results_summary])
    print(f"Average R-squared: {avg_r2:.4f}")
print("="*30)

HEME_pdbs = ["1YZP", "1CRI", "4NVA", "4D3T", "3QM8", "2J18", "1VXA", "1PHA", "7RKR",
             "1DVE", "1DS4", "1SOG", "3P6N", "3WEC", "1D3S", "3FKG", "7PQ1", "3P6U", "8EWQ", "4NVN"]
FAD_pdbs  = ["8A1H", "1REO", "1DOB", "8Z45", "3NLC", "3FG2", "3AXB", "2AQJ", "8Z3G", "1PXC", "8JDY",
             "8Z44", "6RTM", "1OGI", "7OR2", "5GRT", "1E1N", "1E1L", "1OWP", "2YYM"]
ZN_2_pdbs = ["7G3B", "2H2I", "1FT7", "7Y2E", "4CXO", "3T87", "3PB9", "2WEG", "7UHI",
             "2E0P", "7HUX", "7XJ4", "6KM5", "1YSO", "6UGR", "5NS5", "1HFC", "1H3N", "2AW1", "3ORJ"]
CU_1_pdbs = ["2CAL", "1DZ0", "3QQX", "3F7L", "3DSO", "2FT7", "4MAI", "1OF0", "4DPC", "5SSZ", "5NQM",
             "3NT0", "1JZG", "4BTE", "2XV2", "1BXA", "1A8Z", "5MSZ", "2CAK", "7OG7"]

class_map = {}
for p in HEME_pdbs: class_map[p.lower()] = "HEME"
for p in FAD_pdbs:  class_map[p.lower()] = "FAD"
for p in ZN_2_pdbs: class_map[p.lower()] = "Zn2+"
for p in CU_1_pdbs: class_map[p.lower()] = "Cu+"

final_data = []
for pdb, r2 in results_summary:
    # Look up the class, default to "Unknown" if not in your lists
    protein_class = class_map.get(pdb.lower(), "Unknown")
    final_data.append({
        "PDB_Code": pdb.lower(),
        "Class": protein_class,
        "R_Squared": round(r2, 4)
    })

df_final = pd.DataFrame(final_data) # Create DataFrame from the list of dictionaries

# 4. Sort by Class then by R_Squared (highest to lowest)
df_final = df_final.sort_values(by=["Class", "R_Squared"], ascending=[True, False])

# 5. Save the final DataFrame to CSV
csv_path = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all/pLDDT_vs_RMSD_summary.csv"
df_final.to_csv(csv_path, index=False) # Save without the index column

print(f"CSV summary successfully saved to: {csv_path}")



