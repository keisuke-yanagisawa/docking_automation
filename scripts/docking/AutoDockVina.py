import os
from typing import List

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
try:
  from openbabel import pybel # openbabel 3.0.0
except ImportError:
  import pybel #openbabel 2.4

from .DockingBase import DockingBase, DockingRegion
from ..utilities.util import dump_enter_exit_on_debug_log
from ..executables import autodock


@dump_enter_exit_on_debug_log
def rdmol2obmol(rdmol: rdkit.Chem.rdchem.Mol) -> pybel.Molecule:
  """
  RDKitのMoleculeオブジェクトをOpenBabelのMoleculeオブジェクトに変換する
  """
  tmpfile=".tmp.sdf"
  print(Chem.MolToMolBlock(rdmol), file=open(tmpfile, "w"))
  mol = next(pybel.readfile("sdf", tmpfile))
  os.remove(tmpfile)
  return mol

class AutoDockVina(DockingBase):

  def __init__(self, **kwargs):
    self.parameters = self.default_parameters
    self.parameters.update(kwargs)

  @property
  def default_parameters(self) -> dict:
    return {
      "exhaustiveness": 1,
      "n_poses": 100,
    }


  @dump_enter_exit_on_debug_log
  def prepare_ligands(self, ligandfile):
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
    self.ligandfiles = [prep.run()]
    os.remove(tmpfile)

  @dump_enter_exit_on_debug_log
  def prepare_receptor(self, proteinfile):
    """
    ドッキング計算前のタンパク質処理
    #TODO: ドッキングツールによって事前処理は異なるはず。
          ドッキングツールに応じたクラスのメソッドにしたほうがよいか。
    """
    prep = autodock.PrepareReceptor()
    prep.set_receptor(proteinfile)
    self.receptorfile = prep.run()

  @dump_enter_exit_on_debug_log
  def prepare_docking(self, binding_sites: List[DockingRegion]):
    """
    一般には事前にgridを生成するが、AutoDock Vinaはそれが不要。
    そのためこの関数はbinding_sitesを保存するだけになっている
    """
    # TODO 複数箇所のドッキングに対応させる
    self.binding_sites = binding_sites[0:1]

  @dump_enter_exit_on_debug_log
  def dock(self):
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
    if hasattr(self.ligandfiles, "__iter__"):
      if len(self.ligandfiles) > 1:
        logger.warning("multiple ligands are not supported yet. use the first one.")
      ligandfile = self.ligandfiles[0]

    from vina import Vina
    self.v = Vina(seed=42)
    self.v.set_receptor(self.receptorfile)
    self.v.set_ligand_from_file(ligandfile)


    for site in self.binding_sites:
      self.v.compute_vina_maps(center=site.center, box_size=site.width)
      self.v.dock(exhaustiveness=self.parameters["exhaustiveness"], 
                  n_poses=self.parameters["n_poses"])
      self.v.optimize()

  @dump_enter_exit_on_debug_log
  def get_results(self, type: str="pybel"):
    """
    AutoDock Vinaの計算結果を任意の形式に変換して出力する。
    AutoDock Vinaの標準的な入出力はpdbqtファイルだが、
    あまり一般的なフォーマットではないので、異なる形式に変更することを前提としている。
    また、出力された構造には docking_score プロパティが付与されている。
    """

    tmppath = ".tmp.pdbqt"
    self.v.write_poses(tmppath, overwrite=True, n_poses=100)
    scores = [lst[0] for lst in self.v.energies()]
    mols = list(pybel.readfile("pdbqt", tmppath))
    for mol, score in zip(mols, scores):
      mol.data["docking_score"] = score
    os.remove(tmppath)
    return mols
