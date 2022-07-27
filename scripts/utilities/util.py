import os

def getabsolutepath(path):
  path = expandpath(path)
  if not path.startswith("/"): # relative path
    path = os.getcwd() + "/" + path
  return path

def expandpath(path):
  path = os.path.expanduser(path)
  return os.path.expandvars(path)