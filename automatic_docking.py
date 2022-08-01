import argparse
import os
from dataclasses import dataclass
from typing import List

from scripts.utilities import util
from scripts.utilities.util import dump_enter_exit_on_debug_log
from scripts.utilities.logger import logger
from scripts.executables import autodock

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

@dataclass
class BindingSite:
  center: List[float]
  width: List[float]

@dump_enter_exit_on_debug_log
def predict_binding_site(proteinfile: str, margin: float=5) -> List[BindingSite]:
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
    g = Grid((pocket_indices==c_idx).astype(int), [xbin, ybin, zbin])
    g.export(f"test{c_idx}.dx")
    x_indices, y_indices, z_indices = np.where(pocket_indices==c_idx)
    x_min, x_max = xbin[min(x_indices)], xbin[max(x_indices)]
    y_min, y_max = ybin[min(y_indices)], ybin[max(y_indices)]
    z_min, z_max = zbin[min(z_indices)], zbin[max(z_indices)]
    print(x_min, x_max, y_min, y_max, z_min, z_max)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    x_width = x_max - x_min + margin[0]
    y_width = y_max - y_min + margin[1]
    z_width = z_max - z_min + margin[2]
    binding_sites.append(BindingSite([x_center, y_center, z_center], [x_width, y_width, z_width]))

  return binding_sites


@dump_enter_exit_on_debug_log
def prepare_protein(proteinfile):
  """
  ドッキング計算前のタンパク質処理
  #TODO: ドッキングツールによって事前処理は異なるはず。
         ドッキングツールに応じたクラスのメソッドにしたほうがよいか。
  """
  prep = autodock.PrepareReceptor()
  prep.set_receptor(proteinfile)
  pdbqtfile = prep.run()
  return pdbqtfile

@dump_enter_exit_on_debug_log
def rdmol2obmol(rdmol):
  """
  RDKitのMoleculeオブジェクトをOpenBabelのMoleculeオブジェクトに変換する
  """
  tmpfile=".tmp.sdf"
  print(Chem.MolToMolBlock(rdmol), file=open(tmpfile, "w"))
  mol = next(pybel.readfile("sdf", tmpfile))
  os.remove(tmpfile)
  return mol

@dump_enter_exit_on_debug_log
def prepare_ligand(ligandfile):
  """
  ドッキング計算前の化合物処理
  #TODO: ドッキングツールによって事前処理は異なるはず。
  """
  tmpfile=".tmp.mol2"
  smiles = open(ligandfile).read().strip()
  rdmol = Chem.MolFromSmiles(smiles)
  rdmol = Chem.AddHs(rdmol)
  AllChem.EmbedMolecule(rdmol, randomSeed=42)
  obmol = rdmol2obmol(rdmol)
  obmol.write("mol2", tmpfile, overwrite=True)

  # TODO 非芳香環はflexibility考えて複数構造生成した方がいいかも

  # dimorphite-DL と gypsum-DL を入れてあげる必要あり。最初は無視するけど。
  # https://git.durrantlab.pitt.edu/jdurrant/gypsum_dl/-/tree/1.2.0
  # https://git.durrantlab.pitt.edu/jdurrant/dimorphite_dl/-/tree/1.2.4

  prep = autodock.PrepareLigand()
  prep.set_ligand(tmpfile)
  pdbqtfile = prep.run()
  os.remove(tmpfile)
  return [pdbqtfile]


@dump_enter_exit_on_debug_log
def exec_docking(proteinfile: str, ligandfiles: List[str], 
                 binding_sites: List[BindingSite],
                 exhaustiveness: int=1, n_poses: int=100):
  """
  AutoDock Vinaを用いてドッキング計算を行う。
  ここで入力するcenter, box_sizeの単位はいずれもAngstromである。

  MEMO: 
  通常exhaustivenessは8であるが、
  計算時間と計算精度のトレードオフを考えると1のほうが良い、
  ということがREstretto論文執筆時に判明したため
  それを初期設定としている。
  """

  # TODO 複数構造対応
  if hasattr(ligandfiles, "__iter__"):
    if len(ligandfiles) > 1:
      logger.warning("multiple ligands are not supported yet. use the first one.")
    ligandfile = ligandfiles[0]

  from vina import Vina
  v = Vina(seed=42)
  v.set_receptor(proteinfile)
  v.set_ligand_from_file(ligandfile)

  # TODO 複数箇所のドッキングに対応させる
  binding_sites = binding_sites[0:1]
  for site in binding_sites:
    v.compute_vina_maps(center=site.center, box_size=site.width)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    v.optimize()

  return v

@dump_enter_exit_on_debug_log
def output_poses(vina_obj, outputfile):
  """
  AutoDock Vinaの計算結果を任意の形式に変換して出力する。
  AutoDock Vinaの標準的な入出力はpdbqtファイルだが、
  あまり一般的なフォーマットではないので、異なる形式に変更することを前提としている。
  # TODO: exec_docking()以外にドッキングツールに依存する場所があるべきではない。
          あるいは、ドッキングツールはすべて同一のインタフェースを持つclassで書き起こされるべき。
          （こちらのほうが適切に思われる）
  """

  ext = os.path.splitext(os.path.basename(outputfile))[-1][1:]
  if ext == "pdbqt": # pdbqt形式の場合は特別な処理を行わなくて良い
    vina_obj.write_poses(outputfile, overwrite=True, n_poses=100)
    return

  # pybelを通して各種ファイルフォーマットに変換する
  pdbqtfile = ".tmp.pdbqt"
  writer = pybel.Outputfile(ext, outputfile, overwrite=True)
  for mol in pybel.readfile("pdbqt", pdbqtfile):
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

  #TODO: 水分子の除去
  #TODO: kalasantyで複数のポケットを出力する方法
  sites   = predict_binding_site(args.protein)
  protein = prepare_protein(args.protein)
  ligands = prepare_ligand(args.ligand)
  obj     = exec_docking(protein, ligands, sites)
  output_poses(obj, args.output)

  