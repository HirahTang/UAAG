from Bio.PDB import PDBParser
from Bio.PDB import Selection
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import numpy as np
import pickle
import json

bond_type = ['UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE',
 'QUINTUPLE', 'HEXTUPLE',
 'ONEANDAHALF', 'TWOANDAHALF', 'THREEANDAHALF', 'FOURANDAHALF',
 'FIVEANDAHALF', 'AROMATIC', 'IONIC', 'HYDROGEN', 'THREECENTER',
 'DATIVEONE', 'DATIVE', 'DATIVEL', 'DATIVER', 'OTHER', 'ZERO']
BOND_DICT = dict(zip(bond_type, range(len(bond_type))))
hybridization_type = ["UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
HYBRIDIZATION_DICT = dict(zip(hybridization_type, range(len(hybridization_type))))

aa_dict = [
        "Ala", "Asx", "Cys", "Asp", "Glu", "Phe", "Gly", "His", "Ile",
        "Lys", "Leu", "Met", "Asn", "Pro", "Gln", "Arg", "Ser", "Thr",
         "Val", "Trp", "Tyr", "Glx",
        ]

#NAA_DICT = {
#    'MSE': 25, 'HOH': 26, 'NAG': 27, 'FES': 28, ' RB':29 , 'TBA': 30
#    }

#with open(f'data/naa_amino_acid_dict.pkl', 'rb') as f:
#    NAA_DICT = pickle.load(f)
#    f.close()
    
with open('data/naa_amino_acid_dict.json', 'rb') as json_file:
    NAA_DICT = json.load(json_file)
    json_file.close()
    
AA_DICT = [i.upper() for i in aa_dict]
AA_DICT = dict(zip(AA_DICT, range(len(AA_DICT))))


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']

class protein_process:
    def __init__(self, pdb_file, retrieve_feat_mat = False):
        # Read the PDB file
        self.pdb_file = pdb_file
        self.protein = Chem.MolFromPDBFile(pdb_file)
        
        Chem.SanitizeMol(self.protein)
        self.rdmol = Chem.RemoveHs(self.protein)

        self.rd_num_atoms = self.rdmol.GetNumAtoms()
        
        self.feat_mat = np.zeros([self.rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        
        self.ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
    #    self.get_feat_mat()
        
        # Retrieve the atom information
        self.atomic_num = [atom.GetAtomicNum() for atom in self.rdmol.GetAtoms()]
        
        self.aromatic = [atom.GetIsAromatic() for atom in self.rdmol.GetAtoms()]
        self.hybridization = [atom.GetHybridization() for atom in self.rdmol.GetAtoms()]
        self.atomidx = [atom.GetIdx() for atom in self.rdmol.GetAtoms()]
        self.pos = self.rdmol.GetConformer().GetPositions()
        self.atom_name = [atom.GetSymbol() for atom in self.rdmol.GetAtoms()]
        self.atom = self.rdmol.GetAtoms()
        self.bonds = self.rdmol.GetBonds()
        if retrieve_feat_mat:
            self.get_feat_mat()
        
        # Retrieve the atom information relate to the residue
        self.residue = [atom.GetPDBResidueInfo().GetResidueName() for atom in self.rdmol.GetAtoms()]
        self.res_id = [atom.GetPDBResidueInfo().GetResidueNumber() for atom in self.rdmol.GetAtoms()]
        self.get_new_res_id() 
        
        # self.res_id is a list of id for each residue in the protein regardless of the chain 
        # (the id is continuous for disconnected chains)
        self.get_backbone()
        self.centre_of_res_mass = np.zeros([max(self.res_id), 3])
        self.get_centre_of_mass()
        
        self.max_res = max(self.res_id)
        
    def get_centre_of_mass(self):
        ptable = Chem.GetPeriodicTable()

        for res_num in range(max(self.res_id)):
            accum_pos = 0
            accum_mass = 0
            # Take all the atoms in the residue of res_num
            bool_list, atom_index_list = self.get_res_by_num(res_num)
            atom_list = [self.atom[i] for i in atom_index_list]
            pos_list = [self.pos[i] for i in atom_index_list]
            pos_list = np.array(pos_list, dtype=np.float32)
            atom_weight = [ptable.GetAtomicWeight(i.GetAtomicNum()) for i in atom_list]
            for atom_w, atom_p in zip(atom_weight, pos_list):
                accum_pos += atom_w * atom_p
                accum_mass += atom_w
            centre_of_mass = accum_pos/accum_mass

            self.centre_of_res_mass[res_num] = centre_of_mass

    def get_new_res_id(self):
        
        # The res_id is a list of the residue id for each atom in the protein
        # For the original res_id, the id would start over when a new chain is encountered
        # The new res_id is a continuous list of id for all the residues in the protein
        
        """ res_id_new = []
        start = 0
        start_id = -1
        for res in self.res_id:
            if res != start:
                start = res
                start_id += 1
                res_id_new.append(start_id)
            else:
                res_id_new.append(start_id)
        self.res_id = res_id_new """
        
        self.res_id = list(range(len(self.res_id)))
        
    def get_res_by_num(self, res_num):
        bool_list = [id==res_num for id in self.res_id]
        return bool_list, [self.atomidx[i] for i in range(len(self.atom)) if bool_list[i]]
        
    def get_feat_mat(self):
        for feat in self.factory.GetFeaturesForMol(self.rdmol):
            self.feat_mat[feat.GetAtomIds(), self.ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
            
    def get_backbone(self):
        backbone = []
        start = 0
        count = 0
        for id in self.res_id:
            if id == start:
                count += 1
            else:
                start = id
                count = 0
            if count < 4:
                backbone.append(1)
            else:
                backbone.append(0)
        self.is_backbone = backbone
    
    def get_pocket(self, radius, res_num, method=1):
        # Get the res_id of the residues within the radius of res_num
        bool_list, atom_index_list = self.get_res_by_num(res_num)
        
        # Method 1: Get all other residues within the radius of the BACKBONE atoms of res_num
        
        if method == 1:
            atom_list = [self.atom[i] for i in atom_index_list[:4]]
            coord_list = [self.pos[i] for i in atom_index_list[:4]]
            sel_idx = set()
            for coord in coord_list:
                for i, res in enumerate(self.centre_of_res_mass):
                    distance = np.linalg.norm(res - coord, ord=2)
                    if distance < radius and i not in sel_idx and i != res_num:
                        sel_idx.add(i)
            return sel_idx            
        # Method 2: Get all other residues within the radius of the CENTRE of MASS of res_num
        
        elif method == 2:
            sel_idx = set()
            if res_num < self.centre_res_mass.shape[0]:
                coord = self.centre_of_res_mass[res_num]
                
                for i, res in enumerate(self.centre_of_res_mass):

                    distance = np.linalg.norm(res - coord, ord=2)
                    if distance < radius and i not in sel_idx and i != res_num:
                        sel_idx.add(i)
                
                return sel_idx
            else:
                return sel_idx
            
        
        elif method == 3:
            # Get all other residues within the radius of the Alpha Carbon of res_num
            try:
                atom_list = self.atom[atom_index_list[1]]
                coord = self.pos[atom_index_list[1]]
    
                sel_idx = set()
                for i, res in enumerate(self.centre_of_res_mass):
                    distance = np.linalg.norm(res - coord, ord=2)
                    if distance < radius and i not in sel_idx and i != res_num:
                        sel_idx.add(i)
                return sel_idx
            except:
                return set()
        
        else:
            raise ValueError('Invalid method')
        
     
class protein_process_biopython:
    def __init__(self, pdb_file, parser=PDBParser()):
        """
        Class to Define a biopython object, for the convenience of defining pockets
        -----
        INPUT: PBD file directory
        OUTPUT: protein_process_biopython object
            Attributes:
            self.res_id* = [num_res]: id with the length of num_res, starting from 0 to num_res-1,
            self.aa_type* = [num_res]: aa_type for the residues (Encoded by AA_DICT)
            self.mass_centre* = [num_res, 3]: mass centre for the residues
            self.atom_id* = [num_atom]: id with the length of atom, starting from 0 to num_atom
            self.atom_to_res* = [num_atom]: id with the length of atom, a mapping of atom_id to res_id
            self.atom_type* = [num_atom]: atom_type for the atoms
            self.atom_to_aa* = [num_atom]: the residue type of the atom
            self.atom_pos* = [num_atom, 3]: 3D position of the atom
            self.res* = [num_res]: A list of biopython.residue objects (Exclude the non-standard objects)
        """
        self.pdb_name = pdb_file.split('/')[-1].split('.')[0]
        try:
            self.pdb = parser.get_structure(self.pdb_name, pdb_file)
        except:
            # Defne a value error for fail to load pdb
            raise ValueError("Fail to Load PDB") 
        # Get the first model of the PDB
        self.pdb = self.pdb[0]

        self.atom_type = []
        self.atom_to_res = []
        self.atom_to_aa = []
        self.atom_pos = []
        self.ptable = Chem.GetPeriodicTable()
        self.mass_centre = []
        
        self.get_all_res()
        
    def get_centre_of_mass(self, centre_mass, centre_pos):
        accum_pos, accum_mass = 0, 0
        for atom_w, atom_p in zip(centre_mass, centre_pos):
            accum_pos += atom_w * atom_p
            accum_mass += atom_w
        centre_of_mass = accum_pos/accum_mass
        return centre_of_mass
    
    def get_all_res(self):
        self.res = Selection.unfold_entities(self.pdb, "R")
        self.res = [residue for residue in self.res if residue.get_resname() in AA_DICT]
        self.res_id = list(range(len(self.res)))
        self.aa_type = [AA_DICT[residue.get_resname()] for residue in self.res]
        for residue_id in self.res_id:
            centre_pos = []
            centre_mass = []
            for atom in self.res[residue_id]:
                self.atom_type.append(self.ptable.GetAtomicNumber(atom.element))
                self.atom_to_res.append(residue_id)
                self.atom_to_aa.append(self.aa_type[residue_id])
                self.atom_pos.append(atom.get_coord())
                centre_mass.append(self.ptable.GetAtomicWeight(atom.element))
                centre_pos.append(atom.get_coord())
            self.mass_centre.append(self.get_centre_of_mass(centre_mass, centre_pos))
        self.atom_id = list(range(len(self.atom_type)))
        self.atom_pos = np.array(self.atom_pos)
        
    def get_pocket(self, pocket_res_id, centre_mass=True, radius=10):
        if centre_mass:
            centre_mass_pos = self.mass_centre[pocket_res_id]
        else:
            raise ValueError('Undefined Centre Mass')
        sel_idx = set()
        for i, res_mass_centre in enumerate(self.mass_centre):
            distance = np.linalg.norm(res_mass_centre - centre_mass_pos, ord=2)
            if distance < radius and i not in sel_idx and i != pocket_res_id:
                sel_idx.add(i)
        return sel_idx
    
    def construct_pocket(self, res_id):
        
        pocket_res_id = self.get_pocket(res_id)
    #    print(pocket_res_id)
        pocket_atom_id = [i for i in range(len(self.atom_to_res)) if self.atom_to_res[i] in pocket_res_id]
        ligand_atom_id = [i for i in range(len(self.atom_to_res)) if self.atom_to_res[i] == res_id]
        output = (res_id, 
            {
                "pocket_atom_type": np.array(self.atom_type)[pocket_atom_id],
                "pocket_atom_to_aa": np.array(self.atom_to_aa)[pocket_atom_id],
                "pocket_atom_pos": self.atom_pos[pocket_atom_id],
                "ligand_atom_type": np.array(self.atom_type)[ligand_atom_id],
                "ligand_atom_pos": self.atom_pos[ligand_atom_id]
            }
        )
        
        return output
        
        
        
    def get_full_pocket(self):
        output_full = {}
        for residue in self.res_id:
            residue_id, features = self.construct_pocket(residue)
            residue_id = self.pdb_name + str(residue_id)
            output_full[residue_id] = features
            
        return output_full
            

        
        
        
        
def mol2graph(atom_list, protein):
    """
    INPUT:
    atom_list: [Rdkit.atom]
    protein: Rdkit.mol
    
    OUTPUT:
    The Atom Feature list contains [Atomic Number, Aromatic, Hybridization, Atom Index]
    The Edges list contains the indices of the atoms that are connected
    The Edges feature list contains the bond type
    """
    atom_feature_list = []
    atom_idx_list = []
    edges_list = []
    edges_feature_list = []
    for atom in atom_list:
        atom_idx_list.append(atom.GetIdx())
        # The feature list construction code may need expansion
        atom_feature_list.append([atom.GetAtomicNum(), atom.GetIsAromatic(), atom.GetHybridization(), atom.GetIdx()])
    
    atom_map_dict = {atom.GetIdx(): i for i, atom in enumerate(atom_list)}
    
    for bond in protein.GetBonds():
        if bond.GetBeginAtomIdx() in atom_idx_list and bond.GetEndAtomIdx() in atom_idx_list:

            edges_list.append((atom_map_dict[bond.GetBeginAtomIdx()], atom_map_dict[bond.GetEndAtomIdx()]))

            edges_feature_list.append(bond.GetBondType())
            edges_list.append((atom_map_dict[bond.GetEndAtomIdx()], atom_map_dict[bond.GetBeginAtomIdx()]))
            edges_feature_list.append(bond.GetBondType())
            
    return atom_feature_list, edges_list, edges_feature_list
        
def construct_pockets(ligand, pos, protein, radius=10):
    ptable = Chem.GetPeriodicTable()
    # Input: Ligand: list of atoms; pos: Array of atom coordinates; protein: mol object; radius: int
    # Step 1:
        # get the center of mass of the ligand
    # Step 2:
        # get the protein atoms within the radius of the ligand
    # Step 3:
        # Format the dictionary for pickles, contains the ligand atoms features, 
        # ligand bond features, ligand bond connectivity, and protein pocket atom features.
    pos = np.array(pos, dtype=np.float32)
    accum_pos = 0
    accum_mass = 0
    for atomic_num in range(len(ligand)):
        atom_num = ligand[atomic_num].GetAtomicNum()
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atomic_num]*atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos/accum_mass
    
    
    return center_of_mass


def construct_pocket_dict(protein_p, res_id_ls, res_num):
    
    # Step 1: Get all the atoms from res_num of protein_p
    _, ligand_index = protein_p.get_res_by_num(res_num)
    ligand_atoms = [protein_p.atom[i] for i in ligand_index]
    ligand_pos = [protein_p.pos[i] for i in ligand_index]
    
    ligand_feature_list, ligand_edges_list, ligand_edges_feature_list = mol2graph(ligand_atoms, protein_p.rdmol)
    ligand_edges_feature_list = [str(i) for i in ligand_edges_feature_list]
    
    # Ligand_feature_list: [Atomic Number, Aromatic, Hybridization, Atom Index]
    
    # Step 2: Get the atoms from res_id_ls of protein_p
    pocket_atom_index_ls = []
    for pocket_res_num in res_id_ls:
        _, pocket_atom_index = protein_p.get_res_by_num(pocket_res_num)
        pocket_atom_index_ls += pocket_atom_index
    pocket_atoms = [protein_p.atom[i] for i in pocket_atom_index_ls]
    pocket_pos = [protein_p.pos[i] for i in pocket_atom_index_ls]
    pocket_backbone = [protein_p.is_backbone[i] for i in pocket_atom_index_ls]
    protein_element = [protein_p.atomic_num[i] for i in pocket_atom_index_ls]
    
    protein_atom_to_aa_type = [protein_p.residue[i] for i in pocket_atom_index_ls]
    protein_atom_to_aa_type_map = []
    for ele in protein_atom_to_aa_type:
        if ele in AA_DICT.keys():
            protein_atom_to_aa_type_map.append(AA_DICT[ele])
        else:
            if ele in NAA_DICT.keys():
                protein_atom_to_aa_type_map.append(NAA_DICT[ele])
            else:
                raise ValueError('Invalid residue')

    
    # Step 3: Construct the features for ligand atoms and protein atoms
    
    pocket_save = {
        "protein_pos": np.array(pocket_pos),
        "ligand_pos": np.array(ligand_pos),
        "ligand_element": [i[0] for i in ligand_feature_list],
        "ligand_aromatic": [int(i[1]) for i in ligand_feature_list],
        "ligand_hybridization": [HYBRIDIZATION_DICT[str(i[2])] for i in ligand_feature_list],
        "ligand_index_in_pocket": [i[3] for i in ligand_feature_list],
        "ligand_bond_type": [BOND_DICT[i] for i in ligand_edges_feature_list],
        "ligand_bond_index": ligand_edges_list,
        
        "protein_element": protein_element,
        "protein_is_backbone": pocket_backbone,
        "protein_atom_to_aa_type": protein_atom_to_aa_type_map,
        
        "pdb_name": protein_p.pdb_file,
        "res_num": res_num,
    }
    
    return pocket_save

