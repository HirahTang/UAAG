from pdb_construct import protein_process
from pdb_construct import protein_process_biopython
from pdb_construct import construct_pocket_dict
import os
import pickle
from tqdm import tqdm
import argparse
import json
import numpy as np

pdb_file_list = os.listdir('data/pdb/')

def create_pocket(protein_dir, radius=10, method=3, gen_number=1000, gen_start=0):
#    pdb_file_ls = os.listdir(protein_dir)
   
    with open(os.path.join('data/pdb_list.txt'), 'r') as f:
        pdb_list = f.readlines()
        f.close()
        
    pdb_list = [pdb.strip() for pdb in pdb_list]
    pdb_list = [pdb for pdb in pdb_list if pdb.endswith('.pdb')]
    
   
    for pdb_file in tqdm(pdb_list[gen_start:gen_start+gen_number]):
        if os.path.exists(f'data/pocket/method{str(method)}/{pdb_file.split(".")[0]}.pickle'):
            print("Pocket data already exists for", pdb_file.split(".")[0])
            continue
        pdb_adj = pdb_file.split('.')[0]
        pdb_file_full = os.path.join(protein_dir, pdb_file)
        try:
            protein_p = protein_process(pdb_file_full)
        except:
            continue
        pdb_output_dict = {}
        for res_num in range(protein_p.max_res + 1):
            res_list = protein_p.get_pocket(radius, res_num, method)
            if len(res_list) == 0:
                continue
            pocket_dict = construct_pocket_dict(protein_p, res_list, res_num)
            pdb_output_dict[res_num] = pocket_dict
            
        with open(f'data/pocket/method{str(method)}/{pdb_adj}.pickle', 'wb') as handle:
            pickle.dump(pdb_output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_pocket_new(protein_dir, radius=10):
    output_data = {}
    pdb_file_list = os.listdir(protein_dir)
    if args.size > 0:
        for pdb_file in tqdm(pdb_file_list[:args.size]):
            pdb_path = os.path.join(protein_dir, pdb_file)
            try:
                pdb = protein_process_biopython(pdb_path)
            except:
                continue
            pocket_data = pdb.get_full_pocket()
            output_data.update(pocket_data)
        # save pocket_save_test to json
        with open(f'data/uaag_data_{args.output}.json', 'w') as json_file:
            json.dump(output_data, json_file, cls=NumpyEncoder)
            json_file.close()
    else:
        for pdb_file in tqdm(pdb_file_list):
            pdb_path = os.path.join(protein_dir, pdb_file)
            try:
                pdb = protein_process_biopython(pdb_path)
            except:
                continue
            pocket_data = pdb.get_full_pocket()
            output_data.update(pocket_data)
        # save pocket_save_test to json
        with open(f'data/uaag_data_full.json', 'w') as json_file:
            json.dump(output_data, json_file, cls=NumpyEncoder)
            json_file.close()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create pocket data')
    parser.add_argument('--protein_dir', type=str, default='data/pdb_processed/', help='Directory containing protein pdb files')
    parser.add_argument('--radius', type=int, default=10, help='Radius of pocket')
    parser.add_argument('--method', type=int, default=3, help='Method of pocket construction')
    parser.add_argument('--gen_number', type=int, default=0, help='Number of proteins to generate pocket data for')
    parser.add_argument('--gen_start', type=int, default=0, help='Start index of proteins to generate pocket data for')
    parser.add_argument('--output', type=str, default='full', help='Output file name')
    parser.add_argument('--size', type=int, default=0, help='Generated Data Size, 0 for all PDBs in protein_dir')
    args = parser.parse_args()
    create_pocket_new(args.protein_dir)
#    create_pocket(args.protein_dir, args.radius, args.method, args.gen_number, args.gen_start)