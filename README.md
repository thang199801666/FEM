# Mindlin–Reissner 5-DOF Plate FEM (simple example)

This small example implements a 4-node quadrilateral finite element with 5 DOFs per node
(`u, v, w, theta_x, theta_y`) for a Mindlin–Reissner plate with in-plane membrane + bending + shear.

Usage:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the example:

```bash
python plate_fem.py
```

The default example solves a 1m x 1m steel plate (E=210e9 Pa, nu=0.3, t=0.01 m) with
uniform transverse load `q = 1000 N/m^2` and all four edges clamped. The script prints the
maximum transverse deflection.

Notes:
- This is a simple demonstrator, intended for small/educational meshes. It uses a consistent
  transverse load vector and 2x2 Gauss integration. For production use, refine elements,
  improve locking treatment (selective reduced integration or mixed formulation), and validate
  against analytical/reference solutions.
