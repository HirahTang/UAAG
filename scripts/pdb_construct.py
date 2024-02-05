from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import numpy as np

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
        
        self.centre_of_res_mass = np.zeros([max(self.res_id), 3])
        
        self.get_new_res_id() 
        # self.res_id is a list of id for each residue in the protein regardless of the chain 
        # (the id is continuous for disconnected chains)
        self.get_backbone()
        
    def get_centre_of_mass(self, i):
        
        
    
    def get_new_res_id(self):
        res_id_new = []
        start = 0
        start_id = -1
        for res in self.res_id:
            if res != start:
                start = res
                start_id += 1
                res_id_new.append(start_id)
            else:
                res_id_new.append(start_id)
        self.res_id = res_id_new
        
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

def mol2graph(atom_list, protein):
    """
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