
import pickle
import os
import sys
sys.path.append('/home/qcx679/hantang/UAAG')
import os
pdb_tidy_path = "/home/qcx679/hantang/UAAG/data/pdb_processed/"
pdb_tidy_list = os.listdir(pdb_tidy_path)
from tqdm import tqdm
from Bio.PDB import PDBParser
parser = PDBParser()
aa_dict = [
        "Ala", "Asx", "Cys", "Asp", "Glu", "Phe", "Gly", "His", "Ile",
        "Lys", "Leu", "Met", "Asn", "Pro", "Gln", "Arg", "Ser", "Thr",
        "Val", "Trp", "Tyr", "Glx",
        ]
aa_dict = set([i.upper() for i in aa_dict])
res_name_set = {}
naa_set = {}
empty_file = []
from tqdm import tqdm
from Bio.PDB import PDBParser
parser = PDBParser()
for pdb_tidy in tqdm(pdb_tidy_list):
    pdb_tidy = os.path.join(pdb_tidy_path, pdb_tidy)
    try:
        structure = parser.get_structure("X", pdb_tidy)
    except:
        empty_file.append(pdb_tidy)
        continue
    model = structure[0]
    for chain in model:
        #if len(chain) < 100:
        #    continue
        # check if any res_name in chain is of length < 3
        for res in chain: 
            if len(res.get_resname()) < 3:
                if res.get_resname in res_name_set:
                    append_string = pdb_tidy.split("/")[-1].split('.')[0] + chain.get_id()
                    res_name_set[res.get_resname()].append(append_string)
                else:
                    res_name_set[res.get_resname()] = [pdb_tidy.split("/")[-1].split('.')[0] + chain.get_id()]
                
            elif res.get_resname() not in aa_dict:
                if res.get_resname in naa_set:
                    append_string = pdb_tidy.split("/")[-1].split('.')[0] + chain.get_id() + str(res.get_id()[1])
                    naa_set[res.get_resname()].append(append_string)
                else:
                    naa_set[res.get_resname()] = [pdb_tidy.split("/")[-1].split('.')[0] + chain.get_id() + str(res.get_id()[1])]
                
#        if len([res for res in chain if len(res.get_resname()) < 3]) > 0:
#            print("chain {} has res_name of length < 3".format(chain.get_id()))
#            continue


naa_dict = set(naa_set.keys())
non_aa_dict = set(res_name_set.keys())

import json
with open('naa_amino_acid_dict.json', 'w') as f:
    json.dump(list(naa_dict), f)
    f.close()
with open('non_aa_dict.json', 'w') as f:
    json.dump(list(non_aa_dict), f)
    f.close()