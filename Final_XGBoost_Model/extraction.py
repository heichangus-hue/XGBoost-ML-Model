import os
import json
import pandas as pd
import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.SeqUtils.ProtParam import ProteinAnalysis as IP

# --- 1. CONFIGURATION & PDB LISTS ---
# [Insert your DATA_DIRS, AA_MW, and all PDB lists: HEME_pdbs, FAD_pdbs, etc.]
# (Ensures assign_label(code) works correctly)
DATA_DIRS = [
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all", 
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/2nd_batch_all", 
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors_2"
]

AA_MW = {
    'ALA': 71.08, 'ARG': 156.19, 'ASN': 114.11, 'ASP': 115.09, 'CYS': 103.14,
    'GLN': 128.13, 'GLU': 129.12, 'GLY': 57.05,  'HIS': 137.14, 'ILE': 113.16,
    'LEU': 113.16, 'LYS': 128.17, 'MET': 131.19, 'PHE': 147.18, 'PRO': 97.12,
    'SER': 87.08,  'THR': 101.11, 'TRP': 186.21, 'TYR': 163.18, 'VAL': 99.13
}

# Ensure your full lists are here
HEME_pdbs = ["1YZP", "1CRI", "4NVA", "4D3T", "3QM8", "2J18", "1VXA", "1PHA", "7RKR", "1DVE",
             "1DS4", "1SOG", "3P6N", "3WEC", "1D3S", "3FKG", "7PQ1", "3P6U", "8EWQ", "4NVN",
             "5HLQ", "1JIN", "1BEK", "6HQM", "2J19", "2CYM", "2ACP", "1MKR", "6GEQ", "1IRC",
             "5U5U", "6HQK", "8FDJ", "4NVG", "1GWU", "4ZDY", "3M8M", "1CCC", "1C53", "1H5H",
             "1YYD", "2BLI", "4G8U", "5CMV", "1BES", "4RM4", "1UX8", "6BD7", "5KD1", "5XXI"]

FAD_pdbs = ["8A1H", "1REO", "1DOB", "8Z45", "3NLC", "3FG2", "3AXB", "2AQJ", "8Z3G", "1PXC",
            "8JDY", "8Z44", "6RTM", "1OGI", "7OR2", "5GRT", "1E1N", "1E1L", "1OWP", "2YYM",
            "8CCL", "1TJ1", "1DOC", "9GN5", "3QFS", "3GYJ", "3DJJ", "2BAB", "9DTK", "1QNF",
            "8K41", "9F1Y", "6YRZ", "1OWN", "7RT0", "5KOW", "1E63", "1E39", "1PHH", "3COX",
            "3EF6", "6ICI", "7VJ0", "1DOE", "7C4N", "2XRY", "8Z26", "8JDG", "8X38", "5JCK"]

ZN_2_pdbs = ["7G3B", "2H2I", "1FT7", "7Y2E", "4CXO", "3T87", "3PB9", "2WEG", "7UHI", "2E0P",
             "7HUX", "7XJ4", "6KM5", "1YSO", "6UGR", "5NS5", "1HFC", "1H3N", "2AW1", "3ORJ",
             "7G3R", "2H6S", "1FUA", "7Z0N", "4DEF", "3TGE", "3PN5", "2WHZ", "7VFR", "2E88",
             "7HVJ", "7YHD", "6LD4", "1Z1N", "6V4V", "5NYA", "1HKK", "1H71", "2B5W", "3OY0",
             "3RTT", "6CLD", "6YHE", "1G4J", "6PJV", "3KZZ", "7O2S", "7G68", "7JOB", "6LD1"]

CU_1_pdbs = ["2CAL", "1DZ0", "3QQX", "3F7L", "3DSO", "2FT7", "4MAI", "1OF0", "4DPC", "5SSZ",
             "5NQM", "3NT0", "1JZG", "4BTE", "2XV2", "1BXA", "1A8Z", "5MSZ", "2CAK", "7OG7",
             "5NQN", "2BZC", "1HAW", "2XMW", "2QDW", "2JCW", "2CCW", "1SF3", "4TM7", "1UUY",
             "5KBK", "4DP8", "1RJU", "6QVH", "5KBM", "2IDU", "5SSY", "5ARN", "6L9S", "5NQO",
             "3I9Z", "3F7K", "4DPA", "4F2F", "4P5S", "6R01", "4N3T", "2XV0", "5SSX", "1A3Z"]


cofactorless_pdbs = [
    "2BQQ", "1FHG", "4TTP", "4G2E", "3SYJ", "2O6X", "1YU7", "1UIA", 
    "7IE4", "1HE9", "1GVL", "1WKA", "3R63", "3ZM2", "1FQI", "3JSN", 
    "7ID0", "3RLE", "7W8U", "4WJ1", "1BFG", "2WLW", "7IEG", "5R2I", 
    "4WEI", "2TMY", "3PTW", "5P7D", "1YRV", "1W7B", "6K1Y", "1XAA", 
    "5ZO0", "5RCS", "4OEF", "1KAB", "8C9L", "2FGT", "6J6E", "1SNQ", 
    "5DZ9", "6AO9", "3H2G", "1QG5", "1KWB", "3V75", "5C10", "1SZT", 
    "3ZPJ", "1YHW", "6K5M", "4WFU", "7Y5J", "6CB6", "2Y7N", "6F4M", 
    "5YDN", "3NPO", "4P9L", "1QTO", "3ASD", "4G14", "2YLH", "132L", 
    "6JUI", "4QRZ", "3RFY", "5P1Q", "1MJZ", "3VSR", "1HEP", "5NPT", 
    "6VV4", "4PMD", "1PEN", "3O7K", "5NJM", "3OUV", "9L1R", "6RU3",
    "5P18", "1MN4", "1CNU", "6HYN", "2NWD", "2F9F", "2CG7", "1QMR",
    "6G9P", "1LNS", "5R3B", "6HHN", "7X3H", "4U64", "1KF7", "5GLX",
    "3P2J", "1DST", "1DG3", "1L18", "2BMJ", "2DX1", "4KIA", "5I4W",
    "1CUC", "4YWF", "1YG2", "6AR0", "5P3O", "5Y30", "4U3V", "3OKQ",
    "2BTZ", "3UMF", "5G0Z", "2OBT", "7BAJ", "7Q6B", "1ANF", "6L2A",
    "7A8B", "1T7N", "5XCN", "3P7K", "3B5O", "2OBR", "1SLL", "2B2H",
    "8X8S", "6LYT", "3AA5", "1LMQ", "1L0B", "3GW3", "1L40", "3DQI",
    "7IHZ", "3BVS", "5IJM", "2ICC", "7AHW", "1EY0", "6EHX", "3ZDE",
    "4R1B", "1OEM", "9DD2", "8HEK", "3GS9", "1JAM", "4X1O", "2PW5",
    "7ICB", "5OYR", "5LZ1", "7VKP", "7NMO", "3E0E", "5DDV", "1XEI",
    "5YNL", "1I9Y", "1F1S", "5P6A", "2D27", "6P1F", "2PII", "9JLA",
    "1JI6", "7L6W", "2DYI", "7P25", "1LIB", "3GYW", "2OBS", "3VYC",
    "5P0D", "7IDX", "3EMI", "1TK1", "3FFM", "3DGJ", "1ZXQ", "5P8C",
    "2IQT", "5Y5D", "8P33", "5RDZ", "5P39", "1IIW", "5K2N", "5P07",
    "1UEK", "4QSC", "6EFR", "2F8H", "1TP0", "4ACO", "3GVQ", "2JDW"
]

ALL_PDB_CODES = set([c.upper() for c in (HEME_pdbs + FAD_pdbs + ZN_2_pdbs + CU_1_pdbs + cofactorless_pdbs)])

def assign_label(pdb_code):
    code = pdb_code.upper()
    if code in HEME_pdbs: return 0
    elif code in FAD_pdbs: return 1
    elif code in ZN_2_pdbs: return 2
    elif code in CU_1_pdbs: return 3
    elif code in cofactorless_pdbs: return 4
    return None


def extract_global_charge(cif_path):
    parser = MMCIFParser(QUIET=True)
    amino_acids = {'ARG', 'LYS', 'HIS', 'ASP', 'GLU', 'ALA', 'CYS', 'PHE', 'GLY', 'ILE', 
                   'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR'}
    try:
        structure = parser.get_structure("protein", cif_path)
        charge = 0
        for residue in structure.get_residues():
            resname = residue.get_resname().upper()
            if resname in ['ARG', 'LYS']: charge += 1
            elif resname in ['ASP', 'GLU']: charge -= 1
        return charge
    except: return 0

def extract_frustration_data(pdb_code):
    parent_path = '/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/frustratometeR/for_alphafold/results_frustra'
    target_file = os.path.join(parent_path, pdb_code.lower(), f"{pdb_code.lower()}_configurational.csv")
    if not os.path.exists(target_file): return 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(target_file)
        df.columns = [c.replace('"', '').strip() for c in df.columns]
        f_indices = (df['DecoyEnergy'] - df['NativeEnergy']) / df['SDEnergy']
        return df['NativeEnergy'].mean(), df['DecoyEnergy'].mean(), f_indices.mean()
    except: return 0.0, 0.0, 0.0

# --- 3. PRE-LOADING (CACHE TO RAM) ---
SEQUENCE_FILE_PATHS = [
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all/sequences.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/2nd_batch_all/sequences_2nd_batch.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors/sequences_cofactorless.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors_2/sequences_cofactorless_2.txt"
]

def pre_load_structures(data_dirs):
    parser = MMCIFParser(QUIET=True)
    cache = {}
    
    # Load sequences
    seq_map = {}
    for path in SEQUENCE_FILE_PATHS:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split() 
                    if len(parts) >= 2:
                        # This adds the PDB and Sequence to our master map
                        seq_map[parts[0].upper()] = parts[1]
            print(f"Successfully loaded sequences from: {os.path.basename(path)}")
        else:
            print(f"WARNING: Sequence file not found at {path}")

    print(f"Total unique sequences in master map: {len(seq_map)}")

    for d_path in data_dirs:
        if not os.path.exists(d_path): continue
        valid_dirs = [d for d in os.listdir(d_path) if os.path.isdir(os.path.join(d_path, d)) 
                      and d.upper() in ALL_PDB_CODES]
        print(f"Scanning {d_path}: Found {len(valid_dirs)} matching proteins.")

        for folder_name in valid_dirs:
            pdb_code = folder_name.upper()
            sub_path = os.path.join(d_path, folder_name, "seed-1_sample-0")
            cif_path = os.path.join(sub_path, "model.cif")
            conf_path = os.path.join(sub_path, "confidences.json")
            summary_path = os.path.join(sub_path, "summary_confidences.json")

            if os.path.exists(cif_path) and os.path.exists(conf_path):
                try:
                    # 1. Structural Data
                    structure = parser.get_structure(folder_name, cif_path)
                    ca_atoms = [a for a in structure.get_atoms() if a.get_name() == "CA"]
                    coords = np.array([a.get_coord() for a in ca_atoms])
                    center = np.mean(coords, axis=0) if len(coords) > 0 else None
                    
                    # 1. Load Confidence Data ONCE
                    with open(conf_path, 'r') as f: 
                        conf_data = json.load(f)

                    # 2. Load Summary Data (Ranking Score) ONCE
                    ranking_score = 0.0
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f_sum: 
                            ranking_score = json.load(f_sum).get("ranking_score", 0.0)
                    
                    # 2. Global Biophysics (Calculated ONCE)
                    charge = extract_global_charge(cif_path)
                    n_avg, d_avg, f_idx = extract_frustration_data(folder_name)
                    sequence = seq_map.get(pdb_code, "")
                    pI = IP(sequence).isoelectric_point() if sequence else 0.0

                    if center is not None:
                        cache[pdb_code] = {
                            "ca_atoms": ca_atoms, "center": center, 
                            "pI": pI, "charge": charge,
                            "ranking_score": ranking_score,
                            "n_avg": n_avg, "d_avg": d_avg, "f_idx": f_idx,
                            "avg_global_plddt": np.mean(conf_data.get("atom_plddts", [0])),
                            "avg_global_pae": np.mean(conf_data.get("pae", [0])), 
                            "length": len(ca_atoms)
                        }
                except Exception as e: print(f"Error loading {folder_name}: {e}")
    return cache

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # Load your previously saved Bayesian results
    log_df = pd.read_csv("bayesian_search_log.csv")
    # Get the best trial based on the highest value (F1-score)
    best_trial = log_df.sort_values(by="value", ascending=False).iloc[0]
    
    best_radius = best_trial["params_radius"]
    
    print(f"Loading proteins and extracting features for Radius: {best_radius} Å...")
    PROTEIN_CACHE = pre_load_structures(DATA_DIRS)
    
    best_rows = []
    aa_list = list(AA_MW.keys())

    for code, data in PROTEIN_CACHE.items():
        ns = NeighborSearch(data['ca_atoms'])
        nearby = ns.search(data['center'], best_radius, level='R')
        if not nearby: continue
        
        total_in_sphere = len(nearby)
        counts = {aa: 0 for aa in aa_list}
        weights = {aa: 0.0 for aa in aa_list}
        pocket_contacts, total_seq_sep, total_mw = 0, 0, 0.0

        for res in nearby:
            name = res.get_resname()
            if name in aa_list:
                counts[name] += 1
                dist = max(np.linalg.norm(res["CA"].get_coord() - data['center']), 0.1)
                weights[name] += (1.0 / (dist**2))
                total_mw += AA_MW[name]
                
                # NCD/RCO math
                res_contacts = ns.search(res["CA"].get_coord(), 8.0, level='R')
                res_i = res.get_id()[1]
                for c_res in res_contacts:
                    if res_i != c_res.get_id()[1]:
                        pocket_contacts += 1
                        total_seq_sep += abs(res_i - c_res.get_id()[1])

        entropy = sum([-(c/total_in_sphere)*np.log2(c/total_in_sphere) for c in counts.values() if c > 0])

        row_data = {
            'PDB_Code': code,
            'Label': assign_label(code),
            'Ranking_Score': data['ranking_score'],
            'Isoelectric_Point': data['pI'],
            'Charge': data['charge'],
            'Shannon_Entropy': entropy,
            'Avg_Native': data['n_avg'],
            'Avg_Decoy': data['d_avg'],
            'Avg_F_index': data['f_idx'],
            'Global_pLDDT': data['avg_global_plddt'],
            'Global_PAE': data['avg_global_pae'],
            'Avg_MW': total_mw / total_in_sphere,
            'NCD': pocket_contacts / data['length'],
            'RCO': total_seq_sep / (pocket_contacts * data['length']) if pocket_contacts > 0 else 0
        }
        
        for aa in aa_list:
            row_data[f"Norm_{aa}"] = counts[aa] / total_in_sphere
            row_data[f"Weight_{aa}"] = weights[aa]
            
        best_rows.append(row_data)

    df_best = pd.DataFrame(best_rows)
    df_best.to_csv("Best_Features_Table.csv", index=False)
    print(f"\nSUCCESS: Wrote {len(df_best)} proteins to 'Best_Features_Table.csv'")
