import numpy as np
thisdict = [] 
for i in range(6):
    row = {"id": i, "fruit": "apple" + str(i)}
    thisdict.append(row)

#print(thisdict)

for i in range(6):
    print(thisdict[i]["fruit"])
for i in range(6):
    thisdict[i]["orange"] = i**2   # add new key to each dict

#print(thisdict)

from rdkit import Chem

def randic_index(smiles):
    """
    Calculate the Randic (connectivity) index of a molecule from SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    # test case invalid smile
    if mol is None:
        raise ValueError("Invalid SMILES")

    R = 0.0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom()
        j = bond.GetEndAtom()
        d_i = i.GetDegree()
        d_j = j.GetDegree()
        if d_i == 0 or d_j == 0:
            continue  # skip isolated atoms
        R += 1 / np.sqrt(d_i * d_j)

    return R

# Example usage
smiles_list = ["CC(C)C"]
for smi in smiles_list:
    print(smi, "Randić index:", randic_index(smi))

from rdkit.Chem import GetPeriodicTable
pt = GetPeriodicTable()
m_i = pt.GetAtomicWeight('C')

import numpy as np
from rdkit.Chem import GetPeriodicTable

ptable = GetPeriodicTable()

def compute_rg2(molecule):
    """
    Compute Rg^2 for one molecule entry of your dataset.
    molecule: dict with {'ID': int, 'COORDS': numpy array of dicts with 'element','x','y','z'}
    i need to find a better way to do this dictionary
    """
    coords = molecule[0]["COORDS"]
    
    masses = np.array([ptable.GetAtomicWeight(atom['element']) for atom in coords])
    positions = np.array([[atom['x'], atom['y'], atom['z']] for atom in coords])
    
    total_mass = masses.sum()
    com = np.average(positions, axis=0, weights=masses)
    
    rg2 = np.sum(masses * np.sum((positions - com)**2, axis=1)) / total_mass
    return rg2

atom = [{'charge': -0.318566, 'element': 'C', 'x': -0.1683471021, 'y': 1.3993278158, 'z': 0.0143715182},
       {'charge': -0.330121, 'element': 'N', 'x': -0.1112623047, 'y': -0.0487338104, 'z': -0.0231798894},
       {'charge': 0.204096, 'element': 'C', 'x': 0.5905906339, 'y': -0.5983163726, 'z': -0.9525708031},
       {'charge': -0.229604, 'element': 'C', 'x': 1.3650004389, 'y': 0.0477767764, 'z': -2.0126859753},
       {'charge': 0.019776, 'element': 'C', 'x': 2.0630844968, 'y': -0.6593308164, 'z': -2.9324963001},
       {'charge': -0.334722, 'element': 'N', 'x': 2.0824368765, 'y': -2.0248633284, 'z': -2.9205917213},
       {'charge': -0.107526, 'element': 'C', 'x': 1.379936641, 'y': -2.7196815043, 'z': -1.9562905554},
       {'charge': 0.189704, 'element': 'C', 'x': 0.6707413515, 'y': -2.0560230904, 'z': -1.0203911965},
       {'charge': -0.057544, 'element': 'F', 'x': 0.005955771, 'y': -2.7558187537, 'z': -0.106707284},
       {'charge': 0.102243, 'element': 'H', 'x': 0.8219620946, 'y': 1.8636578868, 'z': 0.1520644373},
       {'charge': 0.102335, 'element': 'H', 'x': -0.6027598189, 'y': 1.834287781, 'z': -0.9007077496},
       {'charge': 0.114792, 'element': 'H', 'x': -0.7953054904, 'y': 1.7123981344, 'z': 0.8540185076},
       {'charge': 0.109071, 'element': 'H', 'x': 1.3845838572, 'y': 1.1272551239, 'z': -2.069336879},
       {'charge': 0.129436, 'element': 'H', 'x': 2.6392771101, 'y': -0.1902309888, 'z': -3.7210447836},
       {'charge': 0.268268, 'element': 'H', 'x': 2.6063987143, 'y': -2.5311423487, 'z': -3.6120571697},
       {'charge': 0.138361, 'element': 'H', 'x': 1.4239934204, 'y': -3.7994728445, 'z': -1.9857687961}]

molecule = [{'ID': 133001}]
molecule[0]["COORDS"]=atom


def mol_wts(df):
    from rdkit.Chem import Descriptors
    wts = []
    for row in df.itertuples(index=False):
        smiles = row.smiles_gdb9
        mol = Chem.MolFromSmiles(smiles)
        wts.append(Descriptors.MolWt(mol))

    return wts
     
data_dir = "/mnt/home/elladki1/CMSE802/Final_Project/cmse802_project/data/raw_data"



