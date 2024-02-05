import pickle
from Bio.PDB import *
import os
from tqdm import tqdm
    
def main():
    with open('sidechainnet_casp7_100.pkl', 'rb') as f:
        data = pickle.load(f)
        
    ID_list = data['train']['ids']
    pdbl = PDBList()
    for id in tqdm(ID_list[:10000]):
        
        id = id.split('_')[0]
        print(id)
        try:
            path = pdbl.retrieve_pdb_file(id, file_format='pdb')
            print(os.path.exists(path))
            print(os.path.exists(f'./data/pdb/{id}.pdb'))
            os.rename(path, f'./data/pdb/{id}.pdb')
            
        except:
            print(f'Failed to download {id}')
            continue
        
        
if __name__ == '__main__':
    main()