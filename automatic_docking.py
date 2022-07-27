import argparse
import os

from scripts.utilities import util
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

def get_current_function_name():
  import inspect
  cur_frame = inspect.currentframe()
  return cur_frame.f_back.f_code.co_name

def predict_binding_site(proteinfile, outputdir="output"):
  logger.info(f"start: {get_current_function_name()}")

  model = UNet.load_model(
    "/kalasanty/data/model_scpdb2017.hdf",
    scale=0.5, max_dist=35, 
    featurizer=Featurizer(save_molecule_codes=False)
  )
  
  protein_ext = os.path.splitext(os.path.basename(proteinfile))[-1][1:]
  mol_protein = next(pybel.readfile(protein_ext, proteinfile))

  density, origin, step = model.pocket_density_from_mol(mol_protein)
  print(density)
  pockets = model.get_pockets_segmentation(density, threshold=1e-10, min_size=1)
  import numpy as np
  numbers = set(list(pockets.reshape(-1)))
  for num in numbers:
    if num == 0:
      continue
    # print(pockets.shape, step, origin)
    # xbin = np.arange(pockets.shape[0]) * step[0] + origin[0]
    # ybin = np.arange(pockets.shape[1]) * step[1] + origin[1]
    # zbin = np.arange(pockets.shape[2]) * step[2] + origin[2]
    # g = Grid((pockets==num).astype(int), [xbin, ybin, zbin])
    # g.export(f"test{num}.dx")

    #これでいい感じなので、各pocketを完全に内包するようなboxを作成して計算を回す
    # でも…thresholdとmin_sizeに対して、緩くしすぎると却ってpocket数が減るのはイメージができない
  # print(pockets)
  # print(np.sum(pockets))
  # os.system(f"""
  #   python $KALASANTY_ROOT/kalasanty/scripts/predict.py \
  #   -f {protein_ext} -i {proteinfile} --output {outputdir}
  # """)
  # protein_dir = os.path.dirname(proteinfile)
  # os.system(f"head {outputdir}/{protein_dir}/pocket0.mol2")

  logger.info(f"end: {get_current_function_name()}")

def prepare_protein(proteinfile):
  logger.info(f"start: {get_current_function_name()}")
  prep = autodock.PrepareReceptor()
  prep.set_receptor(proteinfile)
  pdbqtfile = prep.run()
  logger.info(f"end: {get_current_function_name()}")
  return pdbqtfile

def rdmol2obmol(rdmol):
  tmpfile=".tmp.sdf"
  print(Chem.MolToMolBlock(rdmol), file=open(tmpfile, "w"))
  mol = next(pybel.readfile("sdf", tmpfile))
  os.remove(tmpfile)
  return mol

def prepare_ligand(ligandfile):
  tmpfile=".tmp.mol2"
  logger.info(f"start: {get_current_function_name()}")
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
  logger.info(f"end: {get_current_function_name()}")
  os.remove(tmpfile)
  return [pdbqtfile]

def exec_docking(proteinfile, ligandfiles):
  # TODO 複数構造対応
  ligandfile = ligandfiles[0]

  logger.info(f"start: {get_current_function_name()}")
  from vina import Vina
  v = Vina(seed=42)
  v.set_receptor(proteinfile)
  v.set_ligand_from_file(ligandfile)
  v.compute_vina_maps(center=[0,80,110], box_size=[30,30,30])
  v.dock(exhaustiveness=1, n_poses=100)
  v.optimize()
  logger.info(f"end: {get_current_function_name()}")
  return v

def output_poses(vina_obj, outputfile):
  tmpfile = ".tmp.pdbqt"
  vina_obj.write_poses(tmpfile, overwrite=True, n_poses=100)

  ext = os.path.splitext(os.path.basename(outputfile))[-1][1:]
  writer = pybel.Outputfile(ext, outputfile, overwrite=True)
  for mol in pybel.readfile("pdbqt", tmpfile):
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
  obj     = exec_docking(protein, ligands)
  output_poses(obj, args.output)

  