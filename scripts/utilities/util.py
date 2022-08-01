from .logger import logger
import os

def getabsolutepath(path: str) -> str:
  path = expandpath(path)
  if not path.startswith("/"): # relative path
    path = os.getcwd() + "/" + path
  return path

def expandpath(path: str) -> str:
  path = os.path.expanduser(path)
  return os.path.expandvars(path)

def dump_enter_exit_on_debug_log(func):
  """
  関数の開始・終了をdebugログに出力するための関数デコレーター
  """
  def wrapper(*args, **kwargs):
    logger.debug(f"start: {func.__name__}")
    ret = func(*args, **kwargs)
    logger.debug(f"end: {func.__name__}")
    return ret
  return wrapper