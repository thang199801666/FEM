import numpy as np

def gauss_points_2x2():
    a = 1.0 / np.sqrt(3.0)
    pts = np.array([[-a, -a], [ a, -a], [ a,  a], [-a,  a]])
    wts = np.array([1.0, 1.0, 1.0, 1.0])
    return pts, wts

def shape_functions(xi, eta):
    N = np.zeros(4)
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta)
    dN_dxi = np.array([
        -0.25 * (1.0 - eta),
         0.25 * (1.0 - eta),
         0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
         0.25 * (1.0 + xi),
         0.25 * (1.0 - xi)
    ])
    return N, dN_dxi, dN_deta

def element_stiffness(xy, t, E, nu, q=0.0, use_membrane=True, use_bending=True, use_shear=True):
    # xy: (4,2) node coordinates of quad element
    # 5 DOF per node: [u,v,w,theta_x,theta_y] => 20 DOF element
    ng = 4
    pts, wts = gauss_points_2x2()
    Ke = np.zeros((20,20))
    fe = np.zeros(20)

    # material matrices
    # membrane (plane stress)
    if use_membrane:
        coef_m = E * t / (1.0 - nu**2)
        Dm = coef_m * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    else:
        Dm = np.zeros((3,3))

    # bending
    if use_bending:
        coef_b = E * t**3 / (12.0 * (1.0 - nu**2))
        Db = coef_b * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    else:
        Db = np.zeros((3,3))

    # shear
    if use_shear:
        G = E / (2.0 * (1.0 + nu))
        kappa = 5.0 / 6.0
        Ds = kappa * G * t * np.eye(2)
    else:
        Ds = np.zeros((2,2))

    for i in range(ng):
        xi, eta = pts[i]
        N, dN_dxi, dN_deta = shape_functions(xi, eta)
        # Jacobian
        J = np.zeros((2,2))
        for a in range(4):
            J[0,0] += dN_dxi[a] * xy[a,0]
            J[0,1] += dN_dxi[a] * xy[a,1]
            J[1,0] += dN_deta[a] * xy[a,0]
            J[1,1] += dN_deta[a] * xy[a,1]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f'Non-positive Jacobian detJ={detJ:.3e} for element coordinates {xy}')
        invJ = np.linalg.inv(J)
        # derivatives wrt physical coords
        dN_dx = np.zeros((4,2))
        for a in range(4):
            grad = invJ.dot(np.array([dN_dxi[a], dN_deta[a]]))
            dN_dx[a,0] = grad[0]
            dN_dx[a,1] = grad[1]

        # Build B matrices
        # Membrane Bm (3 x 20) -> u,v DOFs
        Bm = np.zeros((3,20))
        for a in range(4):
            idx_u = 5*a
            idx_v = idx_u + 1
            Bm[0, idx_u]     = dN_dx[a,0]
            Bm[1, idx_v]     = dN_dx[a,1]
            Bm[2, idx_u]     = dN_dx[a,1]
            Bm[2, idx_v]     = dN_dx[a,0]

        # Bending Bb (3 x 20) -> rotations theta_x (idx+3), theta_y (idx+4)
        Bb = np.zeros((3,20))
        for a in range(4):
            idx_thx = 5*a + 3
            idx_thy = 5*a + 4
            Bb[0, idx_thx] = dN_dx[a,0]   # d(theta_x)/dx
            Bb[1, idx_thy] = dN_dx[a,1]   # d(theta_y)/dy
            Bb[2, idx_thx] = dN_dx[a,1]   # d(theta_x)/dy
            Bb[2, idx_thy] = dN_dx[a,0]   # d(theta_y)/dx

        # Shear Bs (2 x 20): gamma_x = theta_x + dw/dx ; gamma_y = theta_y + dw/dy
        Bs = np.zeros((2,20))
        for a in range(4):
            idx_w = 5*a + 2
            idx_thx = 5*a + 3
            idx_thy = 5*a + 4
            Bs[0, idx_thx] = N[a]
            Bs[0, idx_w]   = dN_dx[a,0]
            Bs[1, idx_thy] = N[a]
            Bs[1, idx_w]   = dN_dx[a,1]

        weight = wts[i] * detJ
        # accumulate membrane and bending terms with 2x2 Gauss
        Ke += (Bm.T.dot(Dm).dot(Bm) + Bb.T.dot(Db).dot(Bb)) * weight
        # consistent transverse load contribution (only affects w DOFs)
        if q != 0.0:
            for a in range(4):
                idx_w = 5*a + 2
                fe[idx_w] += N[a] * q * weight

    # Shear: use 1-point reduced integration (center) to avoid shear locking
    if use_shear:
        xi = 0.0; eta = 0.0
        N, dN_dxi, dN_deta = shape_functions(xi, eta)
        J = np.zeros((2,2))
        for a in range(4):
            J[0,0] += dN_dxi[a] * xy[a,0]
            J[0,1] += dN_dxi[a] * xy[a,1]
            J[1,0] += dN_deta[a] * xy[a,0]
            J[1,1] += dN_deta[a] * xy[a,1]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f'Non-positive Jacobian detJ={detJ:.3e} for element coordinates {xy}')
        invJ = np.linalg.inv(J)
        dN_dx = np.zeros((4,2))
        for a in range(4):
            grad = invJ.dot(np.array([dN_dxi[a], dN_deta[a]]))
            dN_dx[a,0] = grad[0]
            dN_dx[a,1] = grad[1]
        Bs_c = np.zeros((2,20))
        for a in range(4):
            idx_w = 5*a + 2
            idx_thx = 5*a + 3
            idx_thy = 5*a + 4
            Bs_c[0, idx_thx] = N[a]
            Bs_c[0, idx_w]   = dN_dx[a,0]
            Bs_c[1, idx_thy] = N[a]
            Bs_c[1, idx_w]   = dN_dx[a,1]
        weight_c = 4.0 * detJ
        Ke += Bs_c.T.dot(Ds).dot(Bs_c) * weight_c

    return Ke, fe
