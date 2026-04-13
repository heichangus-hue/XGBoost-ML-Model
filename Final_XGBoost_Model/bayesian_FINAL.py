import os
import json
import optuna
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.SeqUtils.ProtParam import ProteinAnalysis as IP

DATA_DIRS = [
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all", 
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/2nd_batch_all", 
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors_2",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/HEME_third_batch",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/FAD_third_batch",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/Zn_third_batch_all",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/Cu_1_third_batch_all"
]

AA_MW = {
    'ALA': 71.08, 'ARG': 156.19, 'ASN': 114.11, 'ASP': 115.09, 'CYS': 103.14,
    'GLN': 128.13, 'GLU': 129.12, 'GLY': 57.05,  'HIS': 137.14, 'ILE': 113.16,
    'LEU': 113.16, 'LYS': 128.17, 'MET': 131.19, 'PHE': 147.18, 'PRO': 97.12,
    'SER': 87.08,  'THR': 101.11, 'TRP': 186.21, 'TYR': 163.18, 'VAL': 99.13
}

# Lists of PDB
HEME_pdbs = [
    "1YZP", "1CRI", "4NVA", "4D3T", "3QM8", "2J18", "1VXA", "1PHA", "7RKR", "1DVE",
    "1DS4", "1SOG", "3P6N", "3WEC", "1D3S", "3FKG", "7PQ1", "3P6U", "8EWQ", "4NVN",
    "5HLQ", "1JIN", "1BEK", "6HQM", "2J19", "2CYM", "2ACP", "1MKR", "6GEQ", "1IRC",
    "5U5U", "6HQK", "8FDJ", "4NVG", "1GWU", "4ZDY", "3M8M", "1CCC", "1C53", "1H5H",
    "1YYD", "2BLI", "4G8U", "5CMV", "1BES", "4RM4", "1UX8", "6BD7", "5KD1", "5XXI",
    # Third batch added (150)
    "5KDY", "1JPB", "1BEM", "6LY4", "2MGA", "2EUQ", "2ASN", "1MLO", "6KRF", "1IYN", 
    "5VCE", "6LAA", "8ING", "4O6U", "1H57", "5B85", "3OZU", "1CCJ", "1CCB", "1H5Z", 
    "2AL0", "2CIW", "4ICT", "5EAG", "1BJE", "4UGF", "1VXF", "6E8Q", "5M6K", "6A7I", 
    "3O89", "2AQD", "3VOO", "5B50", "2MGG", "7NQN", "8EKO", "1A6G", "6RQ6", "7LS3", 
    "1NZ5", "5ZZG", "3OZZ", "2XVZ", "1NP4", "1Z8P", "9H1M", "6T0K", "2XKG", "1H5L", 
    "3E55", "1HRM", "3A4G", "7V46", "2Z3U", "5EJX", "2HZ2", "7N14", "1CP4", "6HQR", 
    "4AVD", "4NVF", "1LT0", "9Q4Q", "8S53", "3E2N", "1GEI", "4RLR", "2Q8Q", "7TEF", 
    "5HIW", "5G6B", "8GLZ", "8CCR", "3ABA", "4XV4", "1UBB", "6BDD", "1EHG", "1CPG", 
    "5OMU", "2BOQ", "6VZB", "2OHB", "1GEK", "8A8L", "8DJU", "1IWK", "3X34", "5IQX", 
    "7TSA", "3AWP", "1OFK", "3CCP", "2ZT4", "1YMA", "5UFG", "2IIZ", "6ATJ", "8VWK", 
    "5W58", "5M0N", "1EUP", "5ESL", "5IKG", "1PM1", "4NS2", "6HQQ", "1OXA", "4BLN", 
    "3E4N", "2JJO", "8SQP", "5KDZ", "5XKV", "4UG6", "2AMM", "5WK9", "2WHF", "7UF9", 
    "6U87", "6XAM", "1DP6", "2C7X", "7RL2", "7L3Y", "2VLY", "3IW1", "2J0P", "1ECD", 
    "1YRD", "8R9N", "8ZGO", "4UGR", "8F9H", "6EKZ", "2VKU", "1YWC", "5O1K", "4GRC", 
    "3HDL", "8GDI", "8R9Q", "5L1Q", "4APY", "1MNH", "2HZ3", "2EUU", "6MA8", "4NVL"
]

FAD_pdbs = [
    "8A1H", "1REO", "1DOB", "8Z45", "3NLC", "3FG2", "3AXB", "2AQJ", "8Z3G", "1PXC",
    "8JDY", "8Z44", "6RTM", "1OGI", "7OR2", "5GRT", "1E1N", "1E1L", "1OWP", "2YYM",
    "8CCL", "1TJ1", "1DOC", "9GN5", "3QFS", "3GYJ", "3DJJ", "2BAB", "9DTK", "1QNF",
    "8K41", "9F1Y", "6YRZ", "1OWN", "7RT0", "5KOW", "1E63", "1E39", "1PHH", "3COX",
    "3EF6", "6ICI", "7VJ0", "1DOE", "7C4N", "2XRY", "8Z26", "8JDG", "8X38", "5JCK",
    # Third batch added (150)
    "1B8S", "1GRE", "1N1P", "1TDE", "2AR8", "2I0K", "3DJG", "3GWD", "3QVR", "3T31", 
    "4H4S", "4U63", "5HXI", "5ZW8", "6SW2", "7ORZ", "7XPI", "8JDO", "8PLG", "8Z1J", 
    "8Z6I", "9KKG", "1BJK", "1GRG", "1NHS", "1TDF", "2CIE", "2IJG", "3DJL", "3GYI", 
    "3RP6", "3U5S", "4H4W", "4XXG", "5I39", "5ZYN", "6YS2", "7OSQ", "8B7S", "8JDV", 
    "8PXL", "8Z3D", "9C4P", "9ORN", "1BX1", "1GSN", "1NPX", "1U3D", "2CJC", "2JKC", 
    "3DK9", "3LVB", "3RP7", "3W2H", "4HA6", "4YKG", "5OGX", "6C7S", "7AV4", "7QU3", 
    "8BJY", "8JM1", "8QNB", "8Z3X", "9GXB", "9V6B", "1DOD", "1IQR", "1Q9I", "1V5E", 
    "2DJI", "2MBR", "3E2S", "3NG7", "3SX6", "3ZBU", "4K8D", "4YNU", "5UTH", "6DD6", 
    "7B02", "7V8R", "8C1U", "8JM2", "8QNC", "8Z41", "9HNK", "1E62", "1K0I", "1QGY", 
    "1V5G", "2DKH", "2R0C", "3FW1", "3NH3", "3SXI", "4D03", "4NZH", "4YWO", "5VW4", 
    "6KGQ", "7BR0", "7VJ5", "8C6B", "8JM4", "8UIQ", "8Z4K", "9HNL", "1FDR", "1K0L", 
    "1QUF", "1W35", "2GQW", "2YLS", "3G5S", "3NYE", "3SYI", "4D04", "4OVI", "5BUL", 
    "5WGY", "6LR8", "7KPQ", "7VJ6", "8CCM", "8JU4", "8WKC", "8Z4M", "9IQC", "1FNB", 
    "1M6I", "1R2J", "1WAM", "2GR2", "3D1C", "3GSI", "3O55", "3T0K", "4G1V", "4REK", 
    "5GV7", "5Y7A", "6PVI", "7KPT", "7VJB", "8ER1", "8PL6", "8YQM", "8Z4P", "9KKC"
]

ZN_2_pdbs = [
    "7G3B", "2H2I", "1FT7", "7Y2E", "4CXO", "3T87", "3PB9", "2WEG", "7UHI", "2E0P",
    "7HUX", "7XJ4", "6KM5", "1YSO", "6UGR", "5NS5", "1HFC", "1H3N", "2AW1", "3ORJ",
    "7G3R", "2H6S", "1FUA", "7Z0N", "4DEF", "3TGE", "3PN5", "2WHZ", "7VFR", "2E88",
    "7HVJ", "7YHD", "6LD4", "1Z1N", "6V4V", "5NYA", "1HKK", "1H71", "2B5W", "3OY0",
    "3RTT", "6CLD", "6YHE", "1G4J", "6PJV", "3KZZ", "7O2S", "7G68", "7JOB", "6LD1",
    # Third batch added (150)
    "7G4F", "2HJN", "1FUK", "8ATJ", "4DZ9", "3TTC", "3Q1D", "2WXT", "7Y70", "2EJC", 
    "7HX1", "8AA6", "6LUY", "1Z9Y", "6W12", "5OJH", "1HS6", "1H7P", "2BHB", "3P58", 
    "3RZV", "6CPA", "6YO7", "1G52", "6QSQ", "3L7X", "7OXD", "7G6W", "7KUS", "6LND", 
    "5NXV", "3PB6", "5PMX", "6VDN", "4ELC", "9GU7", "1C2D", "8OGF", "9G38", "3CA2", 
    "7JO3", "5ONY", "4RSY", "4EJM", "3BKQ", "3OHR", "8P95", "4R2J", "2EHT", "2AXR", 
    "5FNL", "2CBB", "4ZX4", "4U9B", "6YYZ", "4BJB", "9GCD", "1KS7", "7TW9", "5SIM", 
    "6IK4", "2PMP", "5FLO", "1W9B", "6OIQ", "4I0G", "7FU4", "7CI9", "5AIJ", "6SP0", 
    "3ISI", "7MHH", "1TBF", "1LMH", "7H4E", "3R4X", "8PRC", "4H2I", "1WKF", "3S1G", 
    "2DYX", "5FNM", "4EL4", "5QPD", "7G3I", "5C3K", "3CZS", "7KZ7", "3EBG", "6I3E", 
    "7TSD", "3TT4", "3D09", "5SLD", "5FNH", "4CA2", "7G4K", "7I43", "6PDB", "3P7S", 
    "7I2I", "4P4E", "8PG5", "8QQA", "1PE7", "3RG3", "9YC2", "1HUG", "9FTQ", "4N5P", 
    "5JT5", "4C4M", "1S0U", "3N4B", "6RQU", "7PBA", "4MZI", "3NKM", "7H2Y", "6BCC", 
    "5JC6", "7G59", "5SG4", "2XEF", "4BM9", "3U47", "8C4W", "6QV5", "6JEB", "4B9P", 
    "8CP7", "6U4T", "5PI7", "6TWA", "5JMX", "5AMH", "3P7P", "2WHK", "6DCH", "5ZGW", 
    "7G7W", "2IMR", "1FXU", "8OGD", "4FRV", "3V3H", "3RC6", "2XB4", "8GR6", "2F94"
]

CU_1_pdbs = [
    "2CAL", "1DZ0", "3QQX", "3F7L", "3DSO", "2FT7", "4MAI", "1OF0", "4DPC", "5SSZ",
    "5NQM", "3NT0", "1JZG", "4BTE", "2XV2", "1BXA", "1A8Z", "5MSZ", "2CAK", "7OG7",
    "5NQN", "2BZC", "1HAW", "2XMW", "2QDW", "2JCW", "2CCW", "1SF3", "4TM7", "1UUY",
    "5KBK", "4DP8", "1RJU", "6QVH", "5KBM", "2IDU", "5SSY", "5ARN", "6L9S", "5NQO",
    "3I9Z", "3F7K", "4DPA", "4F2F", "4P5S", "6R01", "4N3T", "2XV0", "5SSX", "1A3Z",
    # Third batch added (18)
    "1UUX", "2FK2", "2FT8", "2GBA", "2IDS", "2RAC", "3EIM", "3FSA", "3ZDW", "4DP1", 
    "4DP4", "4DP6", "4F2E", "4L05", "5C92", "5KBL", "6EK9", "6RW7"
]

# Cofactorless (200)
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

# Pre-loading (CACHE TO RAM)
SEQUENCE_FILE_PATHS = [
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_job/output_batch_dir_2/1st_batch_all/sequences.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/2nd_batch_all/sequences_2nd_batch.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors/sequences_cofactorless.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/proteins_without_cofactors_2/sequences_cofactorless_2.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/HEME_third_batch/sequences_HEME_third_batch.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/FAD_third_batch/sequences_FAD_third_batch.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/Zn_third_batch_all/sequences_Zn_third_batch.txt",
    "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/Cu_1_third_batch_all/sequences_Cu_third_batch.txt"
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
                    # Structural Data
                    structure = parser.get_structure(folder_name, cif_path)
                    ca_atoms = [a for a in structure.get_atoms() if a.get_name() == "CA"]
                    coords = np.array([a.get_coord() for a in ca_atoms])
                    center = np.mean(coords, axis=0) if len(coords) > 0 else None
                    
                    # Load Confidence Data once
                    with open(conf_path, 'r') as f: 
                        conf_data = json.load(f)

                    # Load Summary Data (Ranking Score) once
                    ranking_score = 0.0
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f_sum: 
                            ranking_score = json.load(f_sum).get("ranking_score", 0.0)
                    
                    # Charge, Isoelectirc Point and Frustration calculated once
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

# Bayesian Optimisation
def objective(trial):
    radius = trial.suggest_float("radius", 10.0, 20.0, step=0.5)
    rows = []
    aa_list = list(AA_MW.keys())

    for code, data in PROTEIN_CACHE.items():
        ns = NeighborSearch(data['ca_atoms'])
        nearby = ns.search(data['center'], radius, level='R')
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
                
                # Sequence Separation
                contacts = ns.search(res["CA"].get_coord(), 8.0, level='R')
                res_i = res.get_id()[1]
                for c_res in contacts:
                    if res_i != c_res.get_id()[1]:
                        pocket_contacts += 1
                        total_seq_sep += abs(res_i - c_res.get_id()[1])

        # Local Entropy
        entropy = sum([-(c/total_in_sphere)*np.log2(c/total_in_sphere) for c in counts.values() if c > 0])

        row_data = {
            'PDB_Code': code, 
            'Ranking_Score': data['ranking_score'],
            'Isoelectric_Point': data['pI'], 'Charge': data['charge'], 
            'Shannon_Entropy': entropy, 'Avg_Native': data['n_avg'], 
            'Avg_Decoy': data['d_avg'], 'Avg_F_index': data['f_idx'], 
            'Global_pLDDT': data['avg_global_plddt'], 'Global_PAE': data['avg_global_pae'], 
            'Avg_MW': total_mw / total_in_sphere, 'NCD': pocket_contacts / data['length'],
            'RCO': total_seq_sep / (pocket_contacts * data['length']) if pocket_contacts > 0 else 0
        }
        for aa in aa_list:
            row_data[f"Norm_{aa}"] = counts[aa] / total_in_sphere
            row_data[f"Weight_{aa}"] = weights[aa]
        rows.append(row_data)

    df_trial = pd.DataFrame(rows)
    df_trial['target'] = df_trial['PDB_Code'].apply(assign_label)
    df_trial = df_trial.dropna(subset=['target']).reset_index(drop=True)

    target_map = {0: 'Heme', 1: 'FAD', 2: 'Zn', 3: 'Cu', 4: 'Cofactorless'}
    print(f"--- Features that affect Radius of {radius} Å for the Spherical Pocket ---")
    for label_idx, label_name in target_map.items(): # Iterate through each class label
        is_class = (df_trial['target'] == label_idx).astype(int) # Binary vector for current class vs rest
        corrs = df_trial.select_dtypes(include=[np.number]).drop(columns=['target']).corrwith(is_class).sort_values(ascending=False) # Correlation of each feature with the current class
        top3, bot3 = corrs.head(3), corrs.tail(3)
        print(f"  {label_name:10} | Best: " + ", ".join([f"{n}({v:.2f})" for n,v in top3.items()])) # Print top 3 features with their correlation values
        print(f"  {'':10} | Worst: " + ", ".join([f"{n}({v:.2f})" for n,v in bot3.items()])) # Print bottom 3 features with their correlation values

    
    y = df_trial['target'].astype(int)
    aa_features = [f"Norm_{aa}" for aa in aa_list] + [f"Weight_{aa}" for aa in aa_list]
    biophys = ['Isoelectric_Point', 'Charge', 'Shannon_Entropy', 'Avg_Native', 'Avg_Decoy', 'Avg_F_index']
    all_possible = biophys + ['Ranking_Score', 'Global_pLDDT', 'Global_PAE', 'Avg_MW', 'NCD', 'RCO'] + aa_features
    
    selected_features = [f for f in all_possible if trial.suggest_categorical(f"use_{f}", [True, False])] 
    if not selected_features: return 0 

    X = df_trial[selected_features]
    params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        # FIX: Use "reg_lambda" as the name, and suggest_float is usually better for this
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        "random_state": 42,
        "tree_method": "hist", 
        "device": "cuda", 
        "n_jobs": 1
    }
        
    try:
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X):
            sw = compute_sample_weight('balanced', y.iloc[train_idx]) # Compute sample weights for the training set
            model = XGBClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=sw) # Fit the model with sample weights
            y_true.append(y.iloc[test_idx].values[0])
            y_pred.append(model.predict(X.iloc[test_idx])[0])
            
        # --- FIXED: ACCURACY AND RETURN OUTSIDE THE LOOP ---
        acc = accuracy_score(y_true, y_pred)
        score = f1_score(y_true, y_pred, average='macro')
        trial.set_user_attr("accuracy", acc)
        print(f"\nTrial {trial.number} Finished | Accuracy: {acc:.4f} | F1: {score:.4f}")
        return score
    
    except Exception as e: 
        print(f"Error in Trial {trial.number}: {e}")
        return 0.0

# Execution
if __name__ == "__main__":
    PROTEIN_CACHE = pre_load_structures(DATA_DIRS)
    if PROTEIN_CACHE:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        best_feature_names = [
            f.replace("use_", "") 
            for f, v in study.best_params.items() 
            if f.startswith("use_") and v is True
        ]

        # Save it explicitly
        with open("best_features.json", "w") as f:
            json.dump(best_feature_names, f)
        
        # FINAL EXPORT
        # [Repeat the 'Best Radius' loop to create df_best and save to CSV]
        print(f"\nOptimisation Finished.")
        print(f"Best Accuracy: {study.best_trial.user_attrs['accuracy']:.4f} | Best F1: {study.best_value:.4f}")
        print(f"Optimal Radius: {study.best_params['radius']} A")
        study.trials_dataframe().to_csv("bayesian_search_log.csv", index=False)

    # --- 1. PREPARE BEST PARAMETERS ---
    best_radius = study.best_params['radius']
    best_rows = []
    aa_list = list(AA_MW.keys())
    
    # Identify which features Optuna actually liked
    # (Extracts names from 'use_FeatureName' params)
    best_feature_names = [f.replace("use_", "") for f, v in study.best_params.items() 
                          if f.startswith("use_") and v is True]

    print(f"\n>> Generating Final Master Table using Radius: {best_radius} Å")
    print(f">> Features optimized as 'Useful': {', '.join(best_feature_names)}")

    # Re-extract all the data in terms of a csv output file
    for code, data in PROTEIN_CACHE.items():
        ns = NeighborSearch(data['ca_atoms'])
        nearby = ns.search(data['center'], best_radius, level='R')
        if not nearby: continue
        
        # Local Calculations
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

        # Local Entropy for this radius
        entropy = sum([-(c/total_in_sphere)*np.log2(c/total_in_sphere) for c in counts.values() if c > 0])

        # Create the full row (Global + Local)
        row_data = {
            'PDB_Code': code,
            'Label': assign_label(code), # 0-4 mapping
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
        
        # Add all AA Norms and Weights
        for aa in aa_list:
            row_data[f"Norm_{aa}"] = counts[aa] / total_in_sphere
            row_data[f"Weight_{aa}"] = weights[aa]
            
        best_rows.append(row_data)

    # Save as pkl file (float64 precision)
    df_best = pd.DataFrame(best_rows)
    df_best['target'] = df_best['PDB_Code'].apply(assign_label)

    payload = {
        "df": df_best,
        "selected_features": best_feature_names,
        "best_params": study.best_params
    }
    with open("best_trial_payload.pkl", "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSUCCESS: Wrote {len(df_best)} proteins to 'best_trial_payload.pkl'")
    
    df_best.to_csv("Best_Features_Table.csv", index=False)
    print(f"SUCCESS: Exported Master Table to 'Best_Features_Table.csv'")

    # Create a summary of the best hyperparameters
    summary_data = {
        "Metric": ["Best Accuracy", "Best Macro F1", "Optimal Radius", "Learning Rate", "Max Depth", "Number of Estimators", "Reg Lambda"], "Value": [
        study.best_trial.user_attrs['accuracy'],
        study.best_value,
        study.best_params['radius'],
        study.best_params['learning_rate'],
        study.best_params['max_depth'],
        study.best_params['n_estimators'],
        study.best_params['reg_lambda']]
    }

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("Best_Hyperparameters_Summary.csv", index=False)

    # Also save the list of 'True' features to a text file for easy reading
    with open("Selected_Features_List.txt", "w") as f:
        for feature in best_feature_names:
            f.write(f"{feature}\n")


