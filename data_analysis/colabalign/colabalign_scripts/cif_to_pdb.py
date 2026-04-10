from Bio.PDB import MMCIFParser, PDBIO

cif_parser = MMCIFParser()
io = PDBIO()

# Process 1cri_model.cif
structure = cif_parser.get_structure("1yzp", "1d3s.cif")
io.set_structure(structure)
io.save("1d3s.pdb")

# Process model_0.cif
structure = cif_parser.get_structure("1d3s", "1d3s_cofactor.cif")
io.set_structure(structure)
io.save("1d3s_cofactor.pdb")

#structure = cif_parser.get_structure("model1", "model_1.cif")
#io.set_structure(structure)
#io.save("model_1.pdb")
