import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.SeqUtils.ProtParam import ProteinAnalysis as IP

TEST_DATA_DIR = ["/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_test_set/output_batch_dir_2/test_set"]

AA_MW = {
    'ALA': 71.08, 'ARG': 156.19, 'ASN': 114.11, 'ASP': 115.09, 'CYS': 103.14,
    'GLN': 128.13, 'GLU': 129.12, 'GLY': 57.05,  'HIS': 137.14, 'ILE': 113.16,
    'LEU': 113.16, 'LYS': 128.17, 'MET': 131.19, 'PHE': 147.18, 'PRO': 97.12,
    'SER': 87.08,  'THR': 101.11, 'TRP': 186.21, 'TYR': 163.18, 'VAL': 99.13
}

def extract_global_charge(cif_path):
    parser = MMCIFParser(QUIET=True)
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
    parent_path = ""
    target_file = os.path.join(parent_path, pdb_code.lower(), f"{pdb_code.lower()}_configurational.csv")
    if not os.path.exists(target_file): return 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(target_file)
        df.columns = [c.replace('"', '').strip() for c in df.columns]
        f_indices = (df['DecoyEnergy'] - df['NativeEnergy']) / df['SDEnergy']
        return df['NativeEnergy'].mean(), df['DecoyEnergy'].mean(), f_indices.mean()
    except: return 0.0, 0.0, 0.0

SEQUENCE_FILE_PATHS = [
        "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_test_set/sequence_extraction/sequences_cofactorless_testing.txt",
        "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_test_set/sequence_extraction/sequences_cu_testing.txt",
        "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_test_set/sequence_extraction/sequences_fad_testing.txt",
        "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_test_set/sequence_extraction/sequences_heme_testing.txt",
        "/mnt/iusers01/fse-ugpgt01/chem02/u28460tc/scratch/alphafold_test_set/sequence_extraction/sequences_zn_testing.txt",
    ]

def pre_load_structures(data_dirs, valid_test_codes, seq_map):
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
                      and d.upper() in valid_test_codes]
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
                    
                    # 2. Charge, Isoelectirc Point and Frustration (Calculated ONCE)
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

def extract_pocket_features(PROTEIN_CACHE, radius):
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
            'Avg_MW': total_mw / total_in_sphere, 'NCD': pocket_contacts / total_in_sphere if total_in_sphere > 0 else 0,
            'RCO': total_seq_sep / (pocket_contacts * total_in_sphere) if pocket_contacts > 0 else 0
        }
        for aa in aa_list:
            row_data[f"Norm_{aa}"] = counts[aa] / total_in_sphere
            row_data[f"Weight_{aa}"] = weights[aa] / total_in_sphere
        rows.append(row_data)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Loading the best model")
    with open("final_xgboost_model.pkl", "rb") as f: # Load data in binary format
        final_model = pickle.load(f)

    with open("best_trial_payload.pkl", "rb") as f:
        payload = pickle.load(f)

    selected_features = payload["selected_features"]
    best_radius = payload["best_params"]["radius"]

    # Extract hyperparameters to show they are present
    learning_rate = payload["best_params"]["learning_rate"]
    max_depth = int(payload["best_params"]["max_depth"])
    estimators = int(payload["best_params"]["n_estimators"])
    reg = payload["best_params"]["reg_lambda"]
    
    print(f"\nHyperparameters used:")
    print(f"Pocket Radius: {best_radius} Å")
    print(f"Learning Rate: {learning_rate}")
    print(f"Max Depth: {max_depth}")
    print(f"Estimators: {estimators}")
    print(f"Reg Lambda:{reg}\n")

    class_names = ['HEME', 'FAD', 'Zn2+', 'Cu+', 'Cofactorless']

    class_file_mapping = {
        "testing_heme": 0, "testing_fad": 1, "testing_zn": 2, "testing_cu": 3, "testing_cofactorless": 4
    }

    # Dynamically extract and assign ground truth tracking arrays across the file manifests
    seq_map = {}
    pdb_to_target = {} # Maps PDB code to its target class based on the sequence files
    valid_test_codes = set()

    for path in SEQUENCE_FILE_PATHS:
        if os.path.exists(path):
            filename = os.path.basename(path).lower()
            if "heme" in filename: detected_class = 0
            elif "fad" in filename: detected_class = 1
            elif "zn" in filename: detected_class = 2
            elif "cu" in filename: detected_class = 3
            elif "cofactorless" in filename: detected_class = 4

            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2: # Ensure we have both PDB code and sequence
                        pdb_upper = parts[0].upper()
                        seq_map[pdb_upper] = parts[1] 
                        valid_test_codes.add(pdb_upper) # Keep track of valid PDB codes for testing
                        if detected_class is not None:
                            pdb_to_target[pdb_upper] = detected_class # Assign target class based on detected class from filename

    PROTEIN_CACHE = pre_load_structures(TEST_DATA_DIR, valid_test_codes, seq_map)

    print(f"Extract the {best_radius} Å Pocket Features for the Test Set")

    test_df = extract_pocket_features(PROTEIN_CACHE, best_radius) # Extract features using the best radius from the payload

    test_df['target'] = test_df['PDB_Code'].map(pdb_to_target)
    test_df = test_df.dropna(subset=['target']).reset_index(drop=True)

    X_test = test_df[selected_features]
    y_test = test_df['target'].astype(int)

    # Single direct evaluation
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)
    
    results_list = []
        
    for i, row in test_df.iterrows():
        results_list.append({
            "PDB_Code": row['PDB_Code'],
            "Actual": class_names[y_test.iloc[i]],
            "Predicted": class_names[y_pred[i]],
            "Correct": int(y_test.iloc[i] == y_pred[i]),
            "Prob_HEME": y_proba[i][0], "Prob_FAD": y_proba[i][1],
            "Prob_Zn2": y_proba[i][2], "Prob_Cu1": y_proba[i][3],
            "Prob_Cofactorless": y_proba[i][4]
        })

    pd.DataFrame(results_list).to_csv("test_set_predictions.csv", index=False)

    # Generate Orange Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Independent Test Set Confusion Matrix (Radius: {best_radius} Å)\n"
              f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%, "
              f"Macro F1: {f1_score(y_test, y_pred, average='macro')*100:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("Independent_Test_Confusion_Matrix.png", dpi=500)
    plt.close()

    print(f"\nClassification Report for Test Set (Radius: {best_radius} Å):")
    print(classification_report(y_test, y_pred, target_names=class_names))

        

    
