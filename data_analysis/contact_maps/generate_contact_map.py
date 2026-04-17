import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import MMCIFParser

def generate_distmap(cif_path, output_png, title):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("model", cif_path)
        coords = []
        for residue in structure.get_residues():
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())
            elif "P" in residue:
                coords.append(residue["P"].get_coord())
        
        if not coords: return
        
        coords = np.array(coords)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        plt.figure(figsize=(8, 6))
        plt.imshow(dist_matrix, cmap='inferno_r')
        plt.colorbar(label='Distance / Å')
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        plt.title(title)
        plt.savefig(output_png, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error processing {cif_path}: {e}")

def process_paired_batch(base_path):
    # Get all subdirectories in the base path
    all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Filter for apo codes (those that don't end in _cofactor)
    apo_codes = [d for d in all_dirs if not d.endswith("_cofactor")]

    for pdb_code in apo_codes:
        # Define paths for Apo and Holo
        apo_dir = os.path.join(base_path, pdb_code)
        holo_dir = os.path.join(base_path, f"{pdb_code}_cofactor")
        
        # Check if the cofactor folder actually exists
        if not os.path.exists(holo_dir):
            print(f"Skipping {pdb_code}: No matching _cofactor folder found.")
            continue

        # File paths
        apo_cif = os.path.join(apo_dir, "seed-1_sample-0", "model.cif")
        holo_cif = os.path.join(holo_dir, "seed-1_sample-0", "model.cif")

        # Generate Apo Map (saved in the apo folder)
        if os.path.exists(apo_cif):
            out_apo = os.path.join(apo_dir, f"{pdb_code}_apo_distmap.png")
            generate_distmap(apo_cif, out_apo, f"Apo-{pdb_code.upper()} Contact Map")

        # Generate Holo Map (saved in the apo folder for easy comparison)
        if os.path.exists(holo_cif):
            out_holo = os.path.join(holo_dir, f"{pdb_code}_holo_distmap.png")
            generate_distmap(holo_cif, out_holo, f"Holo-{pdb_code.upper()} Contact Map")

if __name__ == "__main__":
    base_directory = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all"
    process_paired_batch(base_directory)
