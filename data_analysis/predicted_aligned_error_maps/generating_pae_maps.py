import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_pae_map(json_path, output_png, title):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # AlphaFold 3 outputs the PAE matrix under the 'pae' key
        pae_matrix = np.array(data.get('pae', []))
        
        if pae_matrix.size == 0:
            print(f"No PAE data found in {json_path}")
            return

        plt.figure(figsize=(8, 6))
        # cmap='viridis_r' or 'Greens_r' is often used for confidence
        # vmin=0, vmax=30 is the standard error range in Angstroms
        plt.imshow(pae_matrix, cmap='viridis_r', vmin=0, vmax=30)
        plt.colorbar(label='Predicted Aligned Error / Å')
        
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        plt.title(title)
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_png}")
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")

def process_pae_batch(base_path):
    all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    apo_codes = [d for d in all_dirs if not d.endswith("_cofactor")]

    for pdb_code in apo_codes:
        apo_dir = os.path.join(base_path, pdb_code)
        holo_dir = os.path.join(base_path, f"{pdb_code}_cofactor")
        
        if not os.path.exists(holo_dir):
            continue

        # Paths to the JSON confidence files
        apo_json = os.path.join(apo_dir, "seed-1_sample-0", "confidences.json")
        holo_json = os.path.join(holo_dir, "seed-1_sample-0", "confidences.json")

        # Generate Apo PAE (Saved in the apo folder)
        if os.path.exists(apo_json):
            out_apo = os.path.join(apo_dir, f"{pdb_code}_apo_pae.png")
            generate_pae_map(apo_json, out_apo, f"Apo-{pdb_code.upper()} PAE Map")

        # Generate Holo PAE (Saved in the apo folder for comparison)
        if os.path.exists(holo_json):
            out_holo = os.path.join(apo_dir, f"{pdb_code}_holo_pae.png")
            generate_pae_map(holo_json, out_holo, f"Holo-{pdb_code.upper()} PAE Map")

if __name__ == "__main__":
    base_directory = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all"
    process_pae_batch(base_directory)