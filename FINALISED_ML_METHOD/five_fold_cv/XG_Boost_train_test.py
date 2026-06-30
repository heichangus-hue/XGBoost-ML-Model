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
from sklearn.utils.class_weight import compute_sample_weight
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.SeqUtils.ProtParam import ProteinAnalysis as IP
from sklearn.utils.class_weight import compute_sample_weight

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
    with open("best_trial_payload.pkl", "rb") as f:
        payload = pickle.load(f)

    # --- Load and pool training + test data ---
    seq_map = {}
    pdb_to_target = {}
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
                    if len(parts) >= 2:
                        pdb_upper = parts[0].upper()
                        seq_map[pdb_upper] = parts[1]
                        valid_test_codes.add(pdb_upper)
                        if detected_class is not None:
                            pdb_to_target[pdb_upper] = detected_class

    PROTEIN_CACHE = pre_load_structures(TEST_DATA_DIR, valid_test_codes, seq_map)

    selected_features = payload["selected_features"]
    best_radius = payload["best_params"]["radius"]

    # Extract test set features
    test_df = extract_pocket_features(PROTEIN_CACHE, best_radius)
    test_df['target'] = test_df['PDB_Code'].map(pdb_to_target)
    test_df = test_df.dropna(subset=['target']).reset_index(drop=True)

    # Pool training + test
    train_df = payload["df"].dropna(subset=['target']).reset_index(drop=True)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    X = combined_df[selected_features]
    y = combined_df['target'].astype(int)

    best_params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "learning_rate": payload["best_params"]["learning_rate"],
        "max_depth": int(payload["best_params"]["max_depth"]),
        "n_estimators": int(payload["best_params"]["n_estimators"]),
        "reg_lambda": payload["best_params"]["reg_lambda"],
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
        "n_jobs": 1
    }

    class_names = ['HEME', 'FAD', 'Zn2+', 'Cu+', 'Cofactorless']

    # 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies, fold_f1s = [], []  

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]  

        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        y_pred_fold = model.predict(X_test)
        acc = accuracy_score(y_test_fold, y_pred_fold) 
        f1 = f1_score(y_test_fold, y_pred_fold, average='macro')
        fold_accuracies.append(acc)
        fold_f1s.append(f1)

        print(f"Fold {fold+1} | Accuracy: {acc*100:.2f}% | Macro F1: {f1*100:.2f}%")
        print(classification_report(y_test_fold, y_pred_fold, target_names=class_names))

    print(f"\nMean Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
    print(f"Mean Macro F1: {np.mean(fold_f1s)*100:.2f}% ± {np.std(fold_f1s)*100:.2f}%")

    # Train final model on all pooled data
    print("\nTraining final production model on all pooled data...") # Not entirely useful as I will not use the test set for training... 
    final_weights = compute_sample_weight(class_weight='balanced', y=y)
    five_fold_final_model = XGBClassifier(**best_params)
    five_fold_final_model.fit(X, y, sample_weight=final_weights, verbose=False)

    with open("five_fold_cv_xgboost_model.pkl", "wb") as f:
        pickle.dump(five_fold_final_model, f)