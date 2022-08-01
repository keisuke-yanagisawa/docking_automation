import argparse
import os
from dataclasses import dataclass
from typing import List

from scripts.utilities import util
from scripts.utilities.util import dump_enter_exit_on_debug_log
from scripts.utilities.logger import logger
from scripts.executables import autodock
from scripts import docking
from scripts.docking import DockingRegion

from kalasanty.net import UNet
from tfbio.data import Featurizer
import pybel
from gridData import Grid

from rdkit import Chem
from rdkit.Chem import AllChem

try:
  from openbabel import pybel # openbabel 3.0.0
except ImportError:
  import pybel #openbabel 2.4

VERSION = "0.1.0"

@dump_enter_exit_on_debug_log
def predict_binding_site(proteinfile: str, margin: float=5) -> List[DockingRegion]:
  """
  3D-CNNによる化合物結合部位予測手法 Kalasanty に基づいて
  結合部位を列挙する。
  """
  if not hasattr(margin, "__iter__"):
    margin = [margin, margin, margin]

  model = UNet.load_model(
    "/kalasanty/data/model_scpdb2017.hdf",
    scale=0.5, max_dist=35, 
    featurizer=Featurizer(save_molecule_codes=False)
  )
  
  protein_ext = os.path.splitext(os.path.basename(proteinfile))[-1][1:]
  mol_protein = next(pybel.readfile(protein_ext, proteinfile))

  density, origin, step = model.pocket_density_from_mol(mol_protein)
  # print(density)
  pocket_indices = model.get_pockets_segmentation(density)
  import numpy as np

  candidate_indices = set(list(pocket_indices.reshape(-1)))
  candidate_indices.remove(0)

  binding_sites = []

  for c_idx in candidate_indices:
    xbin = np.arange(pocket_indices.shape[0]) * step[0] + origin[0]
    ybin = np.arange(pocket_indices.shape[1]) * step[1] + origin[1]
    zbin = np.arange(pocket_indices.shape[2]) * step[2] + origin[2]
    # g = Grid((pocket_indices==c_idx).astype(int), [xbin, ybin, zbin])
    # g.export(f"test{c_idx}.dx")
    x_indices, y_indices, z_indices = np.where(pocket_indices==c_idx)
    x_min, x_max = xbin[min(x_indices)], xbin[max(x_indices)]
    y_min, y_max = ybin[min(y_indices)], ybin[max(y_indices)]
    z_min, z_max = zbin[min(z_indices)], zbin[max(z_indices)]
    # print(x_min, x_max, y_min, y_max, z_min, z_max)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    x_width = x_max - x_min + margin[0]
    y_width = y_max - y_min + margin[1]
    z_width = z_max - z_min + margin[2]
    binding_sites.append(DockingRegion([x_center, y_center, z_center], [x_width, y_width, z_width]))

  return binding_sites

@dump_enter_exit_on_debug_log
def output_poses(mols: List[pybel.Molecule], outputfile: str) -> None:
  """
  複数のpybel.Moleculeオブジェクトを1つのファイルに出力する。
  """

  ext = os.path.splitext(os.path.basename(outputfile))[-1][1:]
  writer = pybel.Outputfile(ext, outputfile, overwrite=True)
  for mol in mols:
    writer.write(mol)
  writer.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Perform protein-ligand docking automatically")
  parser.add_argument("-p,--protein", dest="protein", required=True)
  parser.add_argument("-l,--ligand", dest="ligand", required=True)
  parser.add_argument("-o,--output", dest="output", required=True)
  parser.add_argument("-v,--verbose", dest="verbose", action="store_true")
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--version", action="version", version=VERSION)
  args = parser.parse_args()

  if args.debug:
      logger.setLevel("debug")
  elif args.verbose:
      logger.setLevel("info")

  adv = docking.AutoDockVina()

  #TODO: 水分子の除去
  #TODO: kalasantyで複数のポケットを出力する方法
  sites   = predict_binding_site(args.protein)
  adv.prepare_receptor(args.protein)
  adv.prepare_ligands(args.ligand)
  adv.prepare_docking(sites)
  adv.dock()
  mols = adv.get_results()
  output_poses(mols, args.output)

  