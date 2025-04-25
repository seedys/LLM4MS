import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Morgan generator
morgan_generator = GetMorganGenerator(radius=20, fpSize=2048)

# Tanimoto score
def calculate_tanimoto(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = morgan_generator.GetFingerprint(mol1)
    fp2 = morgan_generator.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)