# UAAG
UAAG: A Diffusion model for Unatural Amino Acid Generation

### Data Input (Data Loader output format):
```
protein_atom_name(N: str): the atom names of pocket atoms from PDB. N - pocket size

ligand_filename: nan, ignored

protein_element(tensor([N:int])): The atomic number of atoms in the pocket.

protein_atom_to_aa_type(tensor([N:int])): The amino acid type of atoms in the pocket (integer encoded).

ligand_center_of_mass(tensor([3:float])): The center of mass of the ligand (Cartesian coordinate).
protein_molecule_name: "pocket"

ligand_atom_feature(tensor([M, 8], dtype=bool)): The atom features of the ligand (one-hot encoded), M: the size of the ligand. For each atom, there are 8 features: ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
                 the feature is extracted from
                 ```
                    from rdkit.Chem import ChemicalFeatures
                    ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                        'ZnBinder']
                    ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
                    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
                    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
                    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
                    for feat in factory.GetFeaturesForMol(rdmol):
                        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
                ```

protein_is_backbone(tensor([N], dtype=bool)): Whether the pocket atom is backbone atom.
ligand_bond_feature(tensor([B, 5], dtype=bool)): One hot encoding of each bond in the ligand.
    BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
    }

ligand_bond_index(tensor([2, B], dtype=int)): The index of the two atoms that form the bond.

ligand_bond_type(tensor([B], dtype=int)): The bond type of each bond.
    BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
    }

ligand_nbh_list(Dict{keys:[M], values: [neighbor of M]}): The neighbor list of each atom in the ligand.

ligand_smiles(str): The SMILES string of the ligand (nan).

protein_filename(str): The PDB filename of the protein.

protein_atom_feature(tensor([N, 27], dtype=bool)): Concatenation of one-hot encoder of protein_element, 
protein_atom_to_aa_type, protein_is_backbone.

ligand_atom_feature_full: zip of element_list, hybridization_list and aromatic_list, view line 156, transform.py for details

protein_pos(tensor([N, 3], dtype=float)): The Cartesian coordinate of the pocket atoms.

ligand_pos(tensor([M, 3], dtype=float)): The Cartesian coordinate of the ligand atoms.

ligand_element(tensor([M], dtype=int)): The element of the ligand atoms, number by atomic number.

ligand_hybridization(tensor([M], dtype=str)): The hybridization of the ligand atoms.
```