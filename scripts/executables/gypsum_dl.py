import os
import tempfile
from typing import List
from rdkit import Chem
from scripts.utilities.logger import logger

class GypsumDL():
  """
  与えられたSMILESに対して、gypsum_dlを実行し、
  ionization states や tautomers を生成、SMILESとして返す。

  本当は立体構造を出力しているのだが、
  cis-transの挙動が誤っていることが確認されており、
  立体構造が信用できないのでSMILESを出力させることにした。

  TODO: 
    出力件数が最大5件で固定になっているが、調整できるようにすべき。
  """
  def __init__(self, path_to_exe="/gypsum_dl/run_gypsum_dl.py", verbose=False):
    self.verbose = verbose
    self.exe     = path_to_exe

  def set_smiles(self, smi: str) -> None:
    if len(smi.split("\n")) > 1:
      raise ImplementationError("single smiles is only acceptable")

    self.input_smiles = smi
    if len(smi.split()) > 1:
      self.ligand_name = smi.split()[1]
    else:
      self.ligand_name = ""

  def run(self) -> List[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
      open(f"{tmpdir}/input.smi", "w").write(self.input_smiles)

      comm = [self.exe, "-o", tmpdir, "--source", f"{tmpdir}/input.smi"]
      if self.verbose:
        comm.append("-v")
      os.system(" ".join(comm))

      # first mol is output information, not a virtual molecule
      mols = Chem.SDMolSupplier(f"{tmpdir}/gypsum_dl_success.sdf")

    smis = [f"{Chem.MolToSmiles(mol)} {self.ligand_name}" for mol in mols 
            if mol != None and Chem.MolToSmiles(mol) != ""]
    logger.debug(print(smis))
    smis = list(set(smis)) # remove duplicates
    return smis