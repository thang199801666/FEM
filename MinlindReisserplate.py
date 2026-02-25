import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from elements import element_stiffness

def build_mesh(nx, ny, Lx=1.0, Ly=1.0):
    # structured quad mesh
    nxn = nx + 1
    nyn = ny + 1
    xs = np.linspace(0.0, Lx, nxn)
    ys = np.linspace(0.0, Ly, nyn)
    coords = np.zeros((nxn * nyn, 2))
    for j in range(nyn):
        for i in range(nxn):
            idx = j * nxn + i
            coords[idx, :] = [xs[i], ys[j]]
    # element connectivity (4-node quad, counter-clockwise)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n1 = j*nxn + i
            n2 = n1 + 1
            n3 = n2 + nxn
            n4 = n1 + nxn
            elems.append([n1, n2, n3, n4])
    elems = np.array(elems, dtype=int)
    return coords, elems

def assemble(coords, elems, t, E, nu, q, use_membrane=True, use_bending=True, use_shear=True):
    nnodes = coords.shape[0]
    ndof = nnodes * 5
    # Assemble using COO triplet lists for speed
    rows = []
    cols = []
    data = []
    F = np.zeros(ndof)

    for el in elems:
        xy = coords[el, :]
        Ke, fe = element_stiffness(xy, t, E, nu, q, use_membrane=use_membrane, use_bending=use_bending, use_shear=use_shear)
        # global dof indices for this element
        dofs = []
        for n in el:
            base = int(n) * 5
            dofs += list(range(base, base+5))
        # assemble force
        for i_local, I in enumerate(dofs):
            F[I] += fe[i_local]
        # assemble stiffness into triplet lists
        dofs_arr = np.array(dofs, dtype=int)
        # Expand indices and data
        r = np.repeat(dofs_arr, 20)
        c = np.tile(dofs_arr, 20)
        d = Ke.flatten()
        rows.append(r)
        cols.append(c)
        data.append(d)

    if len(rows) > 0:
        rows = np.hstack(rows)
        cols = np.hstack(cols)
        data = np.hstack(data)
    else:
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        data = np.array([], dtype=float)

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    return K, F

def apply_clamped_bc(K, F, coords, tol=1e-12, active_dofs=None):
    # clamp nodes on boundary (x=0,L or y=0,L)
    nd = coords.shape[0]
    fixed = []
    xmin = coords[:,0].min()
    xmax = coords[:,0].max()
    ymin = coords[:,1].min()
    ymax = coords[:,1].max()
    for i in range(nd):
        x,y = coords[i]
        if abs(x-xmin) < tol or abs(x-xmax) < tol or abs(y-ymin) < tol or abs(y-ymax) < tol:
            base = i*5
            fixed += list(range(base, base+5))
    fixed = np.unique(fixed)
    # Create reduced system by eliminating fixed DOFs (avoids changing sparsity)
    ndof = K.shape[0]
    all_dofs = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_dofs, fixed)

    # If active_dofs mask provided, keep only those DOFs (after removing fixed DOFs)
    if active_dofs is not None:
        active = np.array(active_dofs, dtype=bool)
        if active.size != ndof:
            raise ValueError('active_dofs mask length must equal total DOFs')
        # only keep free DOFs that are active
        free = free[active[free]]

    # extract reduced submatrix and load
    K_reduced = K.tocsr()[free,:][:,free].copy()
    F_reduced = F[free].copy()
    return K_reduced, F_reduced, free, fixed

def solve_system(K, F):
    K = K.tocsc()
    U = spla.spsolve(K, F)
    return U

def analytical_clamped_square(a, q, E, nu, t):
    """Approximate center deflection for a uniformly loaded, fully clamped square plate.

    Uses classical plate theory: w0 = k * q * a^4 / D, with D = E t^3 / (12(1-nu^2)).
    Coefficient k ~ 0.0116 for a clamped square under uniform load (standard reference).
    """
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    k = 0.0116
    return k * q * a**4 / D

def run_example():
    # parameters
    Lx = 1.0
    Ly = 1.0
    nx = 20
    ny = 20
    t = 0.01
    E = 210e9
    nu = 0.3
    q = 1e6

    coords, elems = build_mesh(nx, ny, Lx, Ly)

    def run_case(use_membrane, use_bending, use_shear, label):
        K, F = assemble(coords, elems, t, E, nu, q, 
                        use_membrane=use_membrane, 
                        use_bending=use_bending, 
                        use_shear=use_shear)
        # if membrane disabled, drop u(0) and v(1) DOFs from active set to avoid singular stiffness
        ndof = coords.shape[0] * 5
        if not use_membrane:
            mask = np.ones(ndof, dtype=bool)
            mask[0::5] = False
            mask[1::5] = False
            K_red, F_red, free, fixed = apply_clamped_bc(K, F, coords, active_dofs=mask)
        else:
            K_red, F_red, free, fixed = apply_clamped_bc(K, F, coords)
        # If membrane disabled, the reduced stiffness can be singular. Apply tiny diagonal regularization
        # and retry solves if NaNs appear.
        def try_solve_with_regularization(Kmat, Fvec, max_tries=3):
            Kcur = Kmat.tocsr()
            # estimate matrix scale
            try:
                mscale = float(np.max(np.abs(Kcur.data)))
            except Exception:
                mscale = 0.0
            eps = max(mscale * 1e-12, 1e-16)
            for attempt in range(max_tries):
                if attempt > 0:
                    # increase regularization if previous attempt produced NaNs
                    eps *= 100.0
                if eps > 0.0:
                    Kreg = Kcur + sp.eye(Kcur.shape[0], format='csr') * eps
                else:
                    Kreg = Kcur
                try:
                    U = solve_system(Kreg, Fvec)
                except Exception:
                    U = None
                if U is None or (isinstance(U, np.ndarray) and np.isnan(U).any()):
                    # try again with larger eps
                    continue
                return U
            # final attempt without regularization (will raise or return NaNs)
            try:
                return solve_system(Kcur, Fvec)
            except Exception:
                return np.full(Fvec.shape, np.nan)

        U_red = try_solve_with_regularization(K_red, F_red)
        ndof = coords.shape[0] * 5
        U = np.zeros(ndof)
        U[free] = U_red
        # center
        ix = nx // 2
        iy = ny // 2
        center_idx = iy * (nx + 1) + ix
        w_center = U[center_idx*5 + 2]
        return w_center

    # Full model (membrane + bending + shear)
    w_full = run_case(True, True, True, 'full')

    print('Results (center deflection in m):')
    print(f'  FEM model           : {w_full:.6e}')

    # return last solution's w field for plotting (re-run full to get w array)
    K, F = assemble(coords, elems, t, E, nu, q, 
                    use_membrane=True, 
                    use_bending=True, 
                    use_shear=True)
    K_red, F_red, free, fixed = apply_clamped_bc(K, F, coords)
    U_red = solve_system(K_red, F_red)
    ndof = coords.shape[0] * 5
    U = np.zeros(ndof); U[free] = U_red
    nn = coords.shape[0]
    w = np.zeros(nn)
    for i in range(nn):
        w[i] = U[i*5 + 2]
    return coords, elems, w

def plot_with_pyvista(coords, elems, w, show=True, screenshot=None):
    try:
        import pyvista as pv
    except Exception as e:
        print('pyvista not available; run `pip install pyvista` to enable plotting')
        return
    # use w as z coordinate for surface visualization
    points3d = np.column_stack([coords[:,0], coords[:,1], w])

    # build faces array in PyVista format: [4, i0, i1, i2, i3, 4, ...]
    face_list = []
    for el in elems:
        face = np.array([4, int(el[0]), int(el[1]), int(el[2]), int(el[3])], dtype=np.int64)
        face_list.append(face)
    faces = np.hstack(face_list)

    mesh = pv.PolyData(points3d, faces)
    mesh.point_data['deflection'] = w

    # report deflection range
    print(f"Deflection range (m): min={w.min():.6e}, max={w.max():.6e}")

    p = pv.Plotter(off_screen=bool(screenshot), window_size=(400, 300))
    # use parallel (orthographic) projection for the plotter camera
    try:
        p.renderer.camera.SetParallelProjection(True)
    except Exception:
        try:
            p.camera.SetParallelProjection(True)
        except Exception:
            pass
    # Create a refined/smoothed copy for plotting to emulate Abaqus contour smoothing
    try:
        # subdivide for denser sampling and interpolate scalars onto the refined mesh
        mesh_ref = mesh.subdivide(2, 'linear')
        mesh_ref.point_data['deflection'] = mesh.interpolate(mesh_ref)['deflection']
        # optional Laplacian smoothing to further mimic Abaqus postprocessing
        mesh_ref = mesh_ref.smooth(n_iter=20)
        use_mesh = mesh_ref
    except Exception:
        use_mesh = mesh

    actor = p.add_mesh(use_mesh, scalars='deflection', 
               cmap='jet', 
               show_edges=True, 
               show_scalar_bar=False,
               interpolate_before_map=True,
               smooth_shading=True)
    # Ensure VTK mapper interpolation flag is set (works across PyVista versions)
    try:
        # PyVista returns a vtkActor with a mapper
        actor.mapper.SetInterpolateScalarsBeforeMapping(True)
    except Exception:
        try:
            # Fallback: get mapper via GetMapper()
            actor.GetMapper().SetInterpolateScalarsBeforeMapping(True)
        except Exception:
            pass
    # place vertical scalar bar at top-left
    p.add_scalar_bar(title='deflection (m)', vertical=True, position_x=0.035, position_y=0.32, width=0.08, height=0.35)
    if screenshot:
        img = p.screenshot(screenshot)
        print('Screenshot saved to', screenshot)
        p.close()
    elif show:
        p.show()
    else:
        p.show(auto_close=False)


if __name__ == '__main__':
    # Run example and show interactive PyVista plot directly
    coords, elems, w = run_example()
    plot_with_pyvista(coords, elems, w, show=True)
