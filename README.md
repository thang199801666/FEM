# Mindlin–Reissner 5-DOF Plate FEM (simple example)

This small example implements a 4-node quadrilateral finite element with 5 DOFs per node
(`u, v, w, theta_x, theta_y`) for a Mindlin–Reissner plate with in-plane membrane + bending + shear.

```markdown
# Mindlin–Reissner 5-DOF Plate FEM (simple example)

This repository contains a compact 4-node quadrilateral finite-element demonstrator for
Mindlin–Reissner plates (5 DOFs per node: `u, v, w, theta_x, theta_y`).

Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the built-in example solver:

```bash
python MinlindReisserPlate.py
```

Parsing Abaqus `.inp` files

A small regex-based INP parser has been added: `plate_inp_reader.py`.

- Parse the provided `Plate.inp` and print a short summary:

```bash
python plate_inp_reader.py Plate.inp
```

- From Python you can import the parser and load node/element arrays:

```python
from plate_inp_reader import parse_inp_file, nodes_to_array
data = parse_inp_file('Plate.inp')
ids, coords = nodes_to_array(data['nodes'])
elems = data['elements']  # list of {{'id', 'type', 'conn'}}
```

Notes
- The INP parser is intentionally lightweight and line-oriented (regex based). It reads
  `*Node` and `*Element` blocks reliably for standard, well-formed Abaqus exports but does
  not implement the full INP grammar (materials, sections, sets, complex continuations,
  or advanced keyword parsing).
- `MinlindReisserPlate.py` contains the FEM implementation and a small demo that builds a
  structured mesh; you can adapt it to use nodes/elems from an INP by converting the parsed
  `elements` connectivity into the format the solver expects.

If you'd like, I can:
- Integrate INP-loading into `MinlindReisserPlate.py` so the solver reads `Plate.inp` directly.
- Extend the parser to capture materials, sets, and boundary conditions.

```
