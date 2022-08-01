from abc import ABCMeta, abstractmethod
from typing import List
from dataclasses import dataclass

@dataclass
class DockingRegion:
  center: List[float]
  width: List[float]

class DockingBase(metaclass=ABCMeta):
  """
  Base class for all docking algorithms.
  ドッキングツールによって事前準備も異なるため、多くの関数が準備されている。
  """

  @abstractmethod
  def default_parameters(self) -> dict:
    pass

  @abstractmethod
  def prepare_ligands(self, ligandfiles: List[str]) -> None:
    pass
  
  @abstractmethod
  def prepare_receptor(self, receptorfile: str) -> None:
    pass

  @abstractmethod
  def prepare_docking(self, binding_sites: List[DockingRegion]):
    pass

  @abstractmethod
  def dock(self):
    pass

  @abstractmethod
  def get_results(self, type: str="pybel"):
    pass
  
  