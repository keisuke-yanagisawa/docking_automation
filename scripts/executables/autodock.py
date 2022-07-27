import os

class PrepareReceptor():
  def __init__(self, verbose=False):
    self.verbose = verbose
    None

  def set_receptor(self, receptor_filename):
    self.receptor_filename = receptor_filename

  def set_pdbqt_filename(self, pdbqt_filename):
    self.pdbqt_filename = pdbqt_filename

  def run(self):
    if not hasattr(self, "pdbqt_filename"):
      self.pdbqt_filename = os.path.splitext(self.receptor_filename)[0] + ".pdbqt"
    
    comm = ["prepare_receptor", "-r", self.receptor_filename, "-o", self.pdbqt_filename]
    if self.verbose:
      comm.append("-v")
    os.system(" ".join(comm))
    return self.pdbqt_filename
    

class PrepareLigand():
  def __init__(self, verbose=False):
    self.verbose = verbose
    None

  def set_ligand(self, ligand_filename):
    self.ligand_filename = ligand_filename

  def set_pdbqt_filename(self, pdbqt_filename):
    self.pdbqt_filename = pdbqt_filename

  def run(self):
    if not hasattr(self, "pdbqt_filename"):
      self.pdbqt_filename = os.path.splitext(self.ligand_filename)[0] + ".pdbqt"
    
    comm = ["prepare_ligand", "-l", self.ligand_filename, "-o", self.pdbqt_filename]
    if self.verbose:
      comm.append("-v")
    os.system(" ".join(comm))
    return self.pdbqt_filename
