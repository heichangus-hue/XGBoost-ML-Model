import subprocess
import warnings
from os import cpu_count
from pathlib import Path
from argparse import ArgumentParser
from math import isnan
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from shutil import copy
from collections import defaultdict
from time import time
from tqdm.auto import tqdm as tqdm_auto
import shutil

import pandas as pd
import numpy as np
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio.PDB import PDBParser, PDBIO


def script_args():
    parser = ArgumentParser(
        description='Align pairs of protein structures with and without cofactors.'
    )

    # Required arguments
    parser.add_argument('-i', '--input', type=str, required=True, nargs='+', 
                        help='Path(s) to input PDB files (supports wildcards like *.pdb) or directory.')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Path to output directory.')

    # Optional arguments
    parser.add_argument('-c', '--cores', type=int, default=1, help='Number of CPU cores (Default = 1).')
    parser.add_argument('-u', '--usalign', type=Path, default='USalign', help='Path to USalign executable.')

    return parser.parse_args()


class StructureAligner:
    def __init__(self, input_file: Path, output_file: Path, transform_matrix: np.ndarray, 
                 original_filename: str = None, is_reference: bool = False) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.transform_matrix = transform_matrix.astype(np.float64) if transform_matrix is not None else None
        self.original_filename = original_filename if original_filename else input_file.name
        self.is_reference = is_reference
        
        if not is_reference:
            assert transform_matrix.shape == (3, 4)
        if input_file.suffix not in ('.pdb', '.cif'):
            raise ValueError(f'Invalid file extension: {input_file.suffix}')
        self.file_type = input_file.suffix

    def export_coords(self) -> Path:
        """Export coordinates to CSV (for reference structures, no transformation)"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', self.input_file)

        coords_data = []

        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                for residue in chain:
                    res_name = residue.get_resname()
                    res_id = residue.get_id()[1]

                    for atom in residue:
                        coord = atom.get_coord()
                        coords_data.append({
                            'original_input': self.original_filename,
                            'processed_file': self.input_file.name,
                            'chain': chain_id,
                            'residue': res_name,
                            'res_id': res_id,
                            'atom': atom.get_name(),
                            'x': coord[0],
                            'y': coord[1],
                            'z': coord[2]
                        })

        coords_df = pd.DataFrame(coords_data)
        coords_df.insert(0, 'index', range(len(coords_df)))
        csv_output = self.output_file.parent / f'{self.output_file.stem}_reference_coords.csv'
        coords_df.to_csv(csv_output, index=False)
        
        return csv_output

    def transform_coords(self) -> Path:
        """Transform coordinates and export to CSV"""
        parser = PDBParser(QUIET=True)
        io = PDBIO()
        structure = parser.get_structure('structure', self.input_file)

        rotation_matrix = self.transform_matrix[:, 1:4].reshape((3, 3))
        translation_vector = self.transform_matrix[:, 0].reshape((1, 3))

        coords_data = []

        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                for residue in chain:
                    res_name = residue.get_resname()
                    res_id = residue.get_id()[1]

                    for atom in residue:
                        coord = atom.get_coord()
                        new_coord = np.dot(rotation_matrix, coord) + translation_vector.flatten()
                        atom.set_coord(new_coord)

                        coords_data.append({
                            'original_input': self.original_filename,
                            'processed_file': self.input_file.name,
                            'chain': chain_id,
                            'residue': res_name,
                            'res_id': res_id,
                            'atom': atom.get_name(),
                            'x': new_coord[0],
                            'y': new_coord[1],
                            'z': new_coord[2]
                        })

        coords_df = pd.DataFrame(coords_data)
        coords_df.insert(0, 'index', range(len(coords_df)))
        csv_output = self.output_file.parent / f'{self.output_file.stem}_aligned_coords.csv'
        coords_df.to_csv(csv_output, index=False)
        
        io.set_structure(structure)
        io.save(str(self.output_file))
        return csv_output


class CofactorAlign:
    def __init__(self, args) -> None:
        self.output_path = args.output
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        self.cores = min(args.cores, cpu_count()) if args.cores > 0 else 1
        self.usalign_path = args.usalign
        
        # Collect all PDB files from input arguments
        self.all_pdb_files = self._collect_pdb_files(args.input)
        
        # Find all PDB pairs
        self.pdb_pairs = self._find_pdb_pairs()
        
        if not self.pdb_pairs:
            raise ValueError("No matching PDB pairs found! Expected files like: {code}.pdb and {code}_cofactor.pdb")
        
        print(f"Found {len(self.all_pdb_files)} total PDB files")
        print(f"Found {len(self.pdb_pairs)} PDB pairs to align")

    def _collect_pdb_files(self, input_args):
        """Collect all PDB files from input arguments (handles wildcards and directories)"""
        all_files = []
        
        for input_arg in input_args:
            input_path = Path(input_arg)
            
            # If it's a directory, get all PDB files from it
            if input_path.is_dir():
                all_files.extend(input_path.glob("*.pdb"))
            # If it's a file, add it
            elif input_path.is_file() and input_path.suffix == '.pdb':
                all_files.append(input_path)
            # If it contains wildcards (passed as string from shell), use glob
            elif '*' in input_arg or '?' in input_arg:
                # Get parent directory and pattern
                parent = Path(input_arg).parent if Path(input_arg).parent.exists() else Path('.')
                pattern = Path(input_arg).name
                all_files.extend(parent.glob(pattern))
            else:
                # Try to resolve as a path pattern
                try:
                    all_files.extend(Path('.').glob(input_arg))
                except:
                    print(f"Warning: Could not process input: {input_arg}")
        
        # Remove duplicates and ensure they're all PDB files
        unique_files = list(set([f for f in all_files if f.suffix == '.pdb']))
        return sorted(unique_files)

    def _find_pdb_pairs(self):
        """Find pairs of {pdb_code}.pdb and {pdb_code}_cofactor.pdb from collected files"""
        pairs = []
        
        # Create a mapping of PDB codes to files
        pdb_dict = defaultdict(dict)
        
        for pdb_file in self.all_pdb_files:
            if pdb_file.stem.endswith("_cofactor"):
                # This is a cofactor file
                pdb_code = pdb_file.stem.replace("_cofactor", "")
                pdb_dict[pdb_code]['cofactor'] = pdb_file
            else:
                # This is a base file
                pdb_code = pdb_file.stem
                pdb_dict[pdb_code]['base'] = pdb_file
        
        # Find complete pairs
        for pdb_code, files in pdb_dict.items():
            if 'base' in files and 'cofactor' in files:
                pairs.append({
                    'code': pdb_code,
                    'base': files['base'],
                    'cofactor': files['cofactor']
                })
                print(f"  Paired: {files['base'].name} <-> {files['cofactor'].name}")
            elif 'base' in files:
                print(f"  Warning: No cofactor file found for {pdb_code} (have {files['base'].name})")
            elif 'cofactor' in files:
                print(f"  Warning: No base file found for {pdb_code} (have {files['cofactor'].name})")
        
        return pairs

    def _reverse_transformation_matrix(self, transform_mx_forward):
        """Reverse a transformation matrix"""
        rotation_mx_forward = transform_mx_forward[:, 1:4]
        translate_v_forward = transform_mx_forward[:, 0:1]
        rotation_mx_reverse = rotation_mx_forward.T
        translate_v_reverse = -np.dot(rotation_mx_reverse, translate_v_forward)
        return np.hstack((translate_v_reverse, rotation_mx_reverse))

    def _run_usalign(self, model1, model2):
        """Run USalign on two structures"""
        cmd = [self.usalign_path, model1.as_posix(), model2.as_posix(), 
               '-outfmt', '2', '-m', '-']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout, stderr

    def _parse_usalign_stdout(self, stdout: str):
        """Parse USalign output to extract TM scores and transformation matrix"""
        decoded = stdout.decode('UTF8').split('\n')
        alignment = decoded[1].split('\t')
        tm_1, tm_2 = alignment[2], alignment[3]
        fwd_mx = np.array([[d for d in line.split(' ') if d][1:] 
                          for line in decoded[4:7]]).astype(np.float64)
        rev_mx = self._reverse_transformation_matrix(fwd_mx)
        return tm_1, tm_2, fwd_mx, rev_mx

    def align_pairs(self):
        """Align each cofactor structure to its corresponding base structure"""
        results = []
        
        for pair in tqdm_auto(self.pdb_pairs, desc="Aligning pairs"):
            pdb_code = pair['code']
            base_pdb = pair['base']
            cofactor_pdb = pair['cofactor']
            
            # Create output directory for this pair
            pair_output = self.output_path / pdb_code
            pair_output.mkdir(exist_ok=True, parents=True)
            
            print(f"\nProcessing {pdb_code}...")
            print(f"  Reference (no cofactor): {base_pdb.name}")
            print(f"  Target (with cofactor): {cofactor_pdb.name}")
            
            # Run USalign: align cofactor structure TO base structure
            # base_pdb is the reference (target), cofactor_pdb is what we're aligning (query)
            stdout, stderr = self._run_usalign(cofactor_pdb, base_pdb)
            
            # Parse alignment results
            tm_1, tm_2, fwd_mx, rev_mx = self._parse_usalign_stdout(stdout)
            
            print(f"  TM-score: {tm_1} (cofactor->base), {tm_2} (base->cofactor)")
            
            # Export reference coordinates (base structure - no transformation)
            ref_exporter = StructureAligner(
                input_file=base_pdb,
                output_file=pair_output / base_pdb.name,
                transform_matrix=None,
                original_filename=base_pdb.name,
                is_reference=True
            )
            ref_csv = ref_exporter.export_coords()
            print(f"  Saved reference coords: {ref_csv.name}")
            
            # Transform and export cofactor structure
            aligner = StructureAligner(
                input_file=cofactor_pdb,
                output_file=pair_output / cofactor_pdb.name,
                transform_matrix=fwd_mx,
                original_filename=cofactor_pdb.name
            )
            aligned_csv = aligner.transform_coords()
            print(f"  Saved aligned coords: {aligned_csv.name}")
            
            # Copy original structures to output
            copy(base_pdb, pair_output / base_pdb.name)
            
            results.append({
                'pdb_code': pdb_code,
                'reference_file': base_pdb.name,
                'aligned_file': cofactor_pdb.name,
                'tm_score_fwd': tm_1,
                'tm_score_rev': tm_2,
                'reference_csv': ref_csv.name,
                'aligned_csv': aligned_csv.name
            })
        
        # Save summary
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(self.output_path / 'alignment_summary.csv', index=False)
        print(f"\n\nSummary saved to: {self.output_path / 'alignment_summary.csv'}")
        
        return results


def main():
    start = time()
    args = script_args()
    
    instance = CofactorAlign(args)
    results = instance.align_pairs()
    
    print(f'\n{"="*60}')
    print(f'Alignment complete!')
    print(f'Processed {len(results)} pairs')
    print(f'Total elapsed time: {time() - start:.2f} s')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()
