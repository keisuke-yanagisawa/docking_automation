from rdkit import Chem

def MolFromSmilesWithName(smi: str) -> Chem.rdchem.Mol:
  """
  smi: SMILES string
  """
  mol = Chem.MolFromSmiles(smi)
  if len(smi.split()) > 1:
    mol.SetProp("_Name", smi.split()[1])

  return mol