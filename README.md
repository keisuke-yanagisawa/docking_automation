# docking_automation

## How to run

```bash
python automatic_docking.py \
  -p sample/2hu4_A.pdb \
  -l sample/G39.smi \
  -o result.sdf
```

- `-p`: protein file in PDB format
- `-l`: ligand file in SMILES notation
  - Thus far a ligand file containing **SINGLE** ligand is only acceptable.
- `-o`: output file path