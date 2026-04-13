import random
pdb_string = "..." # Input list of PDB codes as a string

pdb_list = [x.strip() for x in pdb_string.split(",") if x.strip()]
index_to_pdb = {i + 1: pdb for i, pdb in enumerate(pdb_list)}
N = len(pdb_list)
print(f"Total PDB Entries:{N}")

# 200 Picks without replacement (unique indices)
random.seed(42)
unique_picks = random.sample(range(1, N+1),200)
unique_pdbs = [index_to_pdb[i] for i in unique_picks]
print("Unique picks (Index --> PDB):")
for index_to_pdb, pdb in zip(unique_picks, unique_pdbs):
    print(index_to_pdb, pdb)
