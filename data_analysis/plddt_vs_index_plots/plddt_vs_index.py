import os
import json
import matplotlib.pyplot as plt
from Bio.PDB.MMCIFParser import MMCIFParser

# 1. Configuration
base_path = "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all"
parser = MMCIFParser(QUIET=True)

# Generate list of apo codes
all_contents = os.listdir(base_path)
apo_codes = [d for d in all_contents 
             if os.path.isdir(os.path.join(base_path, d)) and "_cofactor" not in d]

print(f"Found {len(apo_codes)} PDB codes to process.")

def generate_plddt_plot(cif_path, json_path, output_path, title_text):
    if not os.path.exists(json_path) or not os.path.exists(cif_path):
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        plddts = data.get('atom_plddts', [])

        structure = parser.get_structure("protein", cif_path)
        atom_indices = [int(atom.get_serial_number()) - 1 for atom in structure.get_atoms()]

        if len(atom_indices) != len(plddts):
            print(f"  [!] Mismatch: {title_text} ({len(atom_indices)} atoms vs {len(plddts)} pLDDTs)")
            return

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(atom_indices, plddts, color='#1f77b4', linewidth=1)
        
        ax.set_title(title_text, fontsize=14)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('pLDDT', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Save the figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig) 
        print(f"  [+] Saved: {output_path}")
        
    except Exception as e:
        print(f"  [X] Error on {title_text}: {e}")

# 2. Main Loop
for pdb_code in apo_codes:
    print(f"\nProcessing PDB: {pdb_code.upper()}")
    
    # Define the destination directory (The Apo folder)
    apo_dir = os.path.join(base_path, pdb_code)
    holo_dir = os.path.join(base_path, f"{pdb_code}_cofactor")
    
    # Process APO
    # Source is in seed subfolder, but Destination (output_path) is now in apo_dir
    apo_sub = os.path.join(apo_dir, "seed-1_sample-0")
    out_path_apo = os.path.join(apo_dir, f"{pdb_code}_plddt_plot.png") # Updated
    
    generate_plddt_plot(
        os.path.join(apo_sub, "model.cif"),
        os.path.join(apo_sub, "confidences.json"),
        out_path_apo,
        f"pLDDT vs Index for Apo-{pdb_code.upper()}"
    )

    # Process HOLO
    if os.path.exists(holo_dir):
        holo_sub = os.path.join(holo_dir, "seed-1_sample-0")
        # Save Holo plot in the Apo directory for easy comparison
        out_path_holo = os.path.join(apo_dir, f"{pdb_code}_cofactor_plddt_plot.png") # Updated
        
        generate_plddt_plot(
            os.path.join(holo_sub, "model.cif"),
            os.path.join(holo_sub, "confidences.json"),
            out_path_holo,
            f"pLDDT vs Index for Holo-{pdb_code.upper()}"
        )
    else:
        print(f"  [-] No cofactor folder for {pdb_code}")

print("\nProcessing complete.")
