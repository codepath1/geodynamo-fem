#Version that starts from initial values
# 1. Assembly optimization
# 2. Solve the Poisson subproblem with a small number of processes
# 3. Checkpointing is not supported
# 4. Partial memory optimization
# 5. Add complex topography, including ellipsoidal and Gaussian-relief topography
# Store boundary markers before modifying the topography

import petsc4py 
from mpi4py import MPI 
from petsc4py import PETSc 
import numpy as np 
import dolfinx 
from dolfinx import default_real_type, fem, io, mesh,default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import (CellDiameter, FacetNormal, TestFunction, TrialFunction, avg,
                 conditional, div, dot, dS, ds, dx, grad, gt, inner, outer,
                 TrialFunctions,TestFunctions,curl,cross)

import ufl 
from dolfinx.io import gmshio,XDMFFile  
from basix.ufl import element,mixed_element   
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector, 
                                 create_vector, set_bc,create_matrix) 
import time 
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix_block, assemble_vector_block, 
                                 create_vector_block, set_bc,create_matrix_block,create_matrix) 
import gc
import basix
import adios4dolfinx as adx 
#####Function definitions


if MPI.COMM_WORLD.Get_rank() ==0:   
    print(f'喊话：记录这个算例是测试 高斯地形的收敛性结果测试',flush=True) 


def Linf_norm(B, sampling_degree=12):
    msh = B.function_space.mesh
    tdim = msh.topology.dim

    X = basix.create_lattice(
        msh.basix_cell(),
        sampling_degree,
        basix.LatticeType.equispaced,
        True
    )

    num_cells = msh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)

    values = fem.Expression(B, X).eval(msh, cells)

    # Vector field: compute the Euclidean norm
    if values.ndim == 3:
        point_values = np.linalg.norm(values, axis=-1)
    # Scalar field
    elif values.ndim == 2:
        point_values = np.abs(values)
    else:
        raise RuntimeError(f"无法识别 Expression 返回形状：{values.shape}")

    local_max = (
        float(np.max(point_values))
        if point_values.size > 0
        else 0.0
    )

    return msh.comm.allreduce(local_max, op=MPI.MAX)

def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))

def norm_L2_B(comm, v):
    # """Compute the L2(Ω)-norm of B""" input is A not B 
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(curl(v), curl(v)) * dx)), op=MPI.SUM))

def norm_ds(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(v, v) * ds)), op=MPI.SUM))

def monitor_residual(ksp, its, rnorm):
    if  MPI.COMM_WORLD.Get_rank() ==0:
        print(f'Iteration {its}, Residual norm {rnorm}', flush=True)
        #print(f'Iteration {its}, Residual norm {rnorm}', flush=True)

def norm_r(x): 
    return (x[0]**2+x[1]**2+x[2]**2)**0.5 

def r_2(x): 
    return (x[0]**2+x[1]**2+x[2]**2) 

def noslip(x): 
    return np.stack((np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1]))) 

def boundT_0(x): 
    values = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType) 
    return values 

def boundT_1(x):
    values = np.ones((1, x.shape[1]), dtype=PETSc.ScalarType)
    return values

def boundA(x):
    values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
    return values

def initial_B(x):
    values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
    Br=x[2]*5/8*(-48*ri*ro+(4*ro+ri*(4+3*ro))*6*norm_r(x)-4*(4+3*(ri+ro))*r_2(x)+9*norm_r(x)**3)/r_2(x)
    Bt=-15/4*(norm_r(x)-ri)*(norm_r(x)-ro)*(3*norm_r(x)-4)*np.sqrt(x[0]**2+x[1]**2)/r_2(x)
    Bf=15/8*np.sin(np.pi*(norm_r(x)-ri))*2*x[2]*np.sqrt(x[0]**2+x[1]**2)/r_2(x)

    values[0]=Br*x[0]/norm_r(x)+Bt*x[2]*x[0]/np.sqrt(x[0]**2+x[1]**2)/norm_r(x)-Bf*x[1]/np.sqrt(x[0]**2+x[1]**2)  
    values[1]=Br*x[1]/norm_r(x)+Bt/np.sqrt(x[0]**2+x[1]**2)/norm_r(x)*x[2]*x[1]+Bf*x[0]/np.sqrt(x[0]**2+x[1]**2)  
    values[2]=Br*x[2]/norm_r(x)-Bt*np.sqrt(x[0]**2+x[1]**2)/norm_r(x)

    return values

def T_x(x):
    return (2*norm_r(x)-ri-ro)

def initial_T(x):
    
    r = norm_r(x)
    x_t = T_x(x) 
    
    poly = 1 - 3 * x_t**2 + 3 * x_t**4 - x_t**6
    
    sin2_theta = (x[0]**2 + x[1]**2) / (r**2 + 1e-12)  # Avoid division by zero
    
    # cos(4phi) = 1 - 8cos^2(phi) + 8cos^4(phi), where cos(phi) = x[0]/sqrt(x[0]^2 + x[1]^2)
    xy_norm = x[0]**2 + x[1]**2 + 1e-12  # Avoid division by zero
    cos_phi = x[0] / np.sqrt(xy_norm)
    cos_4phi = 1 - 8 * cos_phi**2 + 8 * cos_phi**4
    term1 = ro * ri / r - ri
    term2 = 21 / np.sqrt(17920 * np.pi) * poly * sin2_theta**2 * cos_4phi
    
    return term1 + term2

###Simulation parameters  
t_end=0.1
#dt=t_end/num_time_steps=5e-4
dt=1e-4
num_time_steps =t_end/dt
maxit=1000   

####Module 2 parameters
out_interval=1 
Ekman=0.001 
Pm=5
Ra=100   
Pr=1  
ro=20/13  
ri=7/13  
tag="plus"   

# Oliver et al. (2025) Gaussian CMB topography parameters
epsilon = 0.2
theta0 = 2.40
phi0 = 0.0
beta_inverse = 40.0

def apply_gaussian_cmb_topography(msh):
    """Deform the spherical fluid shell without changing mesh topology."""

    if msh.geometry.dim != 3:
        raise RuntimeError(
            "Gaussian CMB topography requires a 3-D mesh."
        )

    coordinates = msh.geometry.x[:, :3]
    radius = np.linalg.norm(coordinates, axis=1)

    nonzero = radius > 1.0e-14

    radial_direction = np.zeros_like(coordinates)
    radial_direction[nonzero] = (
        coordinates[nonzero]
        / radius[nonzero, np.newaxis]
    )

    # Direction of the Gaussian-topography center
    topography_direction = np.array(
        [
            np.sin(theta0) * np.cos(phi0),
            np.sin(theta0) * np.sin(phi0),
            np.cos(theta0),
        ],
        dtype=coordinates.dtype,
    )

    # Squared chord length on the unit sphere
    distance_squared = np.sum(
        (
            radial_direction
            - topography_direction[np.newaxis, :]
        ) ** 2,
        axis=1,
    )

    h_topography = np.exp(
        -beta_inverse * distance_squared
    )

    # Deformed outer-boundary radius
    cmb_radius = ro - epsilon * h_topography

    # The theoretical maximum of the Gaussian function is 1
    if ro - epsilon <= ri:
        raise RuntimeError(
            "Gaussian CMB intersects the ICB: reduce epsilon."
        )

    # Normalized radial coordinate in the original spherical shell:
    # ICB corresponds to 0 and CMB corresponds to 1
    shell_coordinate = np.clip(
        (radius - ri) / (ro - ri),
        0.0,
        1.0,
    )

    # Keep the ICB fixed at ri; displacement increases linearly toward the CMB
    new_radius = (
        ri
        + shell_coordinate * (cmb_radius - ri)
    )

    # Update geometry coordinates
    coordinates[nonzero] = (
        radial_direction[nonzero]
        * new_radius[nonzero, np.newaxis]
    )

##########
csv_dir_u='u_csv'+tag+'/'  
csv_dir_B='B_csv'+tag+'/'  

Vol=4/3*np.pi*(ro**3-ri**3)

comm_n = MPI.COMM_WORLD
rank_n = comm_n.rank
size_n = comm_n.size

#name="hypre_cut.xdmf" 

#name="cp6n_0.24w.xdmf" 
name="big_cut.xdmf" 
if MPI.COMM_WORLD.Get_rank()==0:
    print('开始导入网格')

read_mesh = io.XDMFFile(MPI.COMM_WORLD, name, "r")
msh = read_mesh.read_mesh()
read_mesh.close()

gdim = msh.geometry.dim
tdim = msh.topology.dim
fdim = tdim - 1

# Mark the CMB and ICB on the original spherical mesh. These facet indices
# remain valid after changing msh.geometry.x because the topology is unchanged.
spherical_boundaries = [
    (
        1,
        lambda x: np.isclose(
            x[0]**2 + x[1]**2 + x[2]**2,
            ro**2,
        ),
    ),
    (
        2,
        lambda x: np.isclose(
            x[0]**2 + x[1]**2 + x[2]**2,
            ri**2,
        ),
    ),
]

msh.topology.create_entities(fdim)
msh.topology.create_connectivity(fdim, tdim)

facet_indices = []
facet_markers = []
for marker, locator in spherical_boundaries:
    facets = mesh.locate_entities(msh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(
        np.full(facets.size, marker, dtype=np.int32)
    )

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)

facet_tag = mesh.meshtags(
    msh,
    fdim,
    facet_indices[sorted_facets],
    facet_markers[sorted_facets],
)
facet_tag.name = "facet_tags"

# Count only owned facets to avoid double-counting ghost facets under MPI.
num_owned_facets = msh.topology.index_map(fdim).size_local
outer_facets = facet_tag.find(1)
inner_facets = facet_tag.find(2)
num_outer_local = np.count_nonzero(outer_facets < num_owned_facets)
num_inner_local = np.count_nonzero(inner_facets < num_owned_facets)
num_outer_global = msh.comm.allreduce(num_outer_local, op=MPI.SUM)
num_inner_global = msh.comm.allreduce(num_inner_local, op=MPI.SUM)

if num_outer_global == 0:
    raise RuntimeError("No CMB facets were found on the original mesh.")
if num_inner_global == 0:
    raise RuntimeError("No ICB facets were found on the original mesh.")

if msh.comm.rank == 0:
    print(f"Original CMB facets: {num_outer_global}", flush=True)
    print(f"Original ICB facets: {num_inner_global}", flush=True)

# Apply the Gaussian geometry before creating any function space or
# interpolating any initial condition.
apply_gaussian_cmb_topography(msh)

# The tags still refer to the same topological facets after deformation.
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

#Build data structures required for subsequent communication:
msh_n=msh 
Vn = fem.functionspace(msh_n, ("Lagrange", 1)) 

imap_n = Vn.dofmap.index_map
bs_n = Vn.dofmap.index_map_bs

if bs_n != 1:
    raise RuntimeError("当前程序只针对标量有限元空间，要求 index_map_bs = 1。") 

num_owned_n = imap_n.size_local

local_range_n = imap_n.local_range
if callable(local_range_n):
    local_range_n = local_range_n()

global_start_n, global_end_n = local_range_n

coords_n_all_local = Vn.tabulate_dof_coordinates()
coords_n_owned = coords_n_all_local[:num_owned_n, :gdim].copy()

local_dofs_n_owned = np.arange(num_owned_n, dtype=np.int64)
global_dofs_n_owned = global_start_n + local_dofs_n_owned

###Function spaces  
U_space=fem.functionspace(msh,("CG",2,(gdim,)))  
P_space=fem.functionspace(msh,("CG",1))  
A_space=fem.functionspace(msh,("N1curl",2))   
T_space=fem.functionspace(msh,("CG",1))  

u,v=ufl.TrialFunction(U_space), ufl.TestFunction(U_space)  
P,q=ufl.TrialFunction(P_space), ufl.TestFunction(P_space)  
A,Av=ufl.TrialFunction(A_space), ufl.TestFunction(A_space)  
T,Tv=ufl.TrialFunction(T_space), ufl.TestFunction(T_space)  

u_offset = U_space.dofmap.index_map.size_local * U_space.dofmap.index_map_bs  
p_offset = u_offset + P_space.dofmap.index_map.size_local * P_space.dofmap.index_map_bs   
A_offset = p_offset + A_space.dofmap.index_map.size_local * A_space.dofmap.index_map_bs   
T_offset = A_offset + T_space.dofmap.index_map.size_local * T_space.dofmap.index_map_bs   

# Funcion space for visualising the velocity field
WW = fem.functionspace(msh, ("CG", 2, (gdim,)))
#TT = fem.functionspace(msh, ("Discontinuous Lagrange", k + 1))
TT = fem.functionspace(msh, ("Lagrange", 1))

###Define initial values
bp_file_path = "./out.bp"

u_n_1=fem.Function(U_space)
u_n_2=fem.Function(U_space)
u_D = fem.Function(U_space)

A_n_1=fem.Function(A_space)
A_n_2=fem.Function(A_space)

A_n_1.interpolate(initial_B)
A_n_2.interpolate(initial_B)

T_n_1=fem.Function(T_space)
T_n_2=fem.Function(T_space)
T_n_1.interpolate(initial_T)
T_n_2.interpolate(initial_T)
T_D=fem.Function(T_space)
T_D.interpolate(boundT_1)   #cell_tag_oc.find(12)

class BoundaryCondition_u():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = fem.Function(U_space)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = fem.locate_dofs_topological(U_space, fdim, facets)
            self._bc = fem.dirichletbc(u_D, dofs)
        elif type == "Neumann":
                self._bc = inner(values, v) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(u-values[1], v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type


class BoundaryCondition_A(): ##Magnetic vector-potential boundary
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = fem.Function(A_space)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = fem.locate_dofs_topological(A_space, fdim, facets)
            self._bc = fem.dirichletbc(u_D, dofs)
        elif type == "Neumann":
                self._bc = inner(values, Av) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(A-values[1], Av)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type


class BoundaryCondition_T(): ##Boundary
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = fem.Function(T_space)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = fem.locate_dofs_topological(T_space, fdim, facets)
            self._bc = fem.dirichletbc(u_D, dofs)
        elif type == "Neumann":
                self._bc = inner(values, Tv) * ds(marker)
        elif type == "Robin":
            self._bc = values[0] * inner(T-values[1], Tv)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type


boundary_conditions_u = [BoundaryCondition_u("Dirichlet", 1, noslip),
                       BoundaryCondition_u("Dirichlet",2, noslip)]


boundary_conditions_A = [BoundaryCondition_A("Dirichlet", 1, boundA),
                         BoundaryCondition_A("Dirichlet", 2, boundA)]

boundary_conditions_T = [BoundaryCondition_T("Dirichlet", 1, boundT_0),
                       BoundaryCondition_T("Dirichlet", 2, boundT_1)]
#######

bcs=[] 
for condition in boundary_conditions_u: 
    if condition.type == "Dirichlet": 
        bcs.append(condition.bc) 

for condition in boundary_conditions_A: 
    if condition.type == "Dirichlet": 
        bcs.append(condition.bc)

for condition in boundary_conditions_T: 
    if condition.type == "Dirichlet": 
        bcs.append(condition.bc) 

#solver_comm 
m=32  #Number of subproblem processes; update when tuning parameters
solver_ranks = list(range(m))
is_solver_rank = rank_n < m
color = 0 if is_solver_rank else MPI.UNDEFINED
solve_comm = comm_n.Split(color=color, key=rank_n)

b_m = None
xm = None

if is_solver_rank:

    rank_m = solve_comm.rank
    size_m = solve_comm.size

    read_mesh0 = io.XDMFFile(solve_comm, name, "r") 
    msh_m = read_mesh0.read_mesh() 
    read_mesh0.close() 
    
    # The reduced-communicator Poisson operator must use the same geometry
    # as the main problem. input_global_indices are unchanged by this map.
    apply_gaussian_cmb_topography(msh_m)

    Vm = fem.functionspace(msh_m, ("Lagrange", 1))

    imap_m = Vm.dofmap.index_map
    bs_m = Vm.dofmap.index_map_bs

    num_owned_m = imap_m.size_local

    local_range_m = imap_m.local_range
    if callable(local_range_m):
        local_range_m = local_range_m()

    global_start_m, global_end_m = local_range_m

    coords_m_all_local = Vm.tabulate_dof_coordinates()
    coords_m_owned = coords_m_all_local[:num_owned_m, :gdim].copy()

    local_dofs_m_owned = np.arange(num_owned_m, dtype=np.int64)
    global_dofs_m_owned = global_start_m + local_dofs_m_owned
    
    # Assemble the Poisson matrix
    ds_default = ufl.Measure("ds", domain=msh_m) 
    hm = CellDiameter(msh_m) 
    pm,qm=ufl.TrialFunction(Vm), ufl.TestFunction(Vm)  
    apm=inner(grad(pm), grad(qm)) * dx  
    apm+=0.1/hm*inner(pm,qm)*ds_default  
    Mpm=fem.form(apm)  
    # Matrix and vector
    PPm = assemble_matrix(Mpm, bcs=[])   
    PPm.assemble()     
    Lm = fem.form(inner(fem.Constant(msh_m, 0.0), qm) * dx)
    bm = create_vector(Lm)  

    xm = bm.duplicate()   
    ksp_gm = PETSc.KSP().create(solve_comm)  # type: ignore   
    ksp_gm.setOperators(PPm)   
    ksp_gm.setType('cg')  #'preonly'   
    ksp_gm.setInitialGuessNonzero(False)     
    ksp_gm.getPC().setType(PETSc.PC.Type.JACOBI)     
    opts = PETSc.Options()       # type: ignore    
    opts["ksp_rtol"] = 1.0e-3    ##Residual tolerance     
    opts["ksp_atol"] = 1.0e-6     # Absolute residual tolerance    
    opts["ksp_max_it"] =2000       # Set the maximum number of iterations to 800    
    ksp_gm.setFromOptions() 
    #ksp_gm.setMonitor(monitor_residual)   
    # Release the mesh and other unnecessary memory

# Define communication functions:
# ============================================================
# Subproblem communication: use input_global_indices from the original mesh M to build the Qm -> Qn mapping
# ============================================================
value_dtype = np.dtype(PETSc.ScalarType)

def _as_row(dofmap_like, i):
    """
    兼容 numpy 二维数组形式和 AdjacencyList.links(i) 形式。
    """
    try:
        return np.asarray(dofmap_like[i], dtype=np.int64)
    except Exception:
        return np.asarray(dofmap_like.links(i), dtype=np.int64)


def owned_cg1_dof_input_ids(msh, V, num_owned, label):
    """
    对标量 CG1/Q1 空间，建立：

        local owned dof id  ->  导入前原始网格 M 中的 input_global_index

    返回数组 input_ids，长度为 num_owned：
        input_ids[local_dof] = 原始 M 中对应顶点/几何点的全局输入编号

    注意：
    1. 这个函数针对标量 Lagrange 1 空间。
    2. 依赖 msh.geometry.input_global_indices。
    3. 对 CG2、N1curl、DG 等空间不能直接用这个函数。
    """
    if V.dofmap.index_map_bs != 1:
        raise RuntimeError(
            f"{label}: 当前映射函数只支持标量空间，要求 index_map_bs = 1，"
            f"但得到 {V.dofmap.index_map_bs}"
        )

    if not hasattr(msh.geometry, "input_global_indices"):
        raise RuntimeError(
            f"{label}: msh.geometry.input_global_indices 不存在。"
            "这个方法需要 XDMF 读入后保留导入前 M 的几何点全局编号。"
        )

    tdim = msh.topology.dim

    # cell -> vertex connectivity
    msh.topology.create_connectivity(tdim, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)

    cell_imap = msh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts

    # geometry dof local id -> input mesh M global geometry id
    input_global_indices = np.asarray(msh.geometry.input_global_indices, dtype=np.int64)

    # cell -> geometry dofs
    geom_dofmap = msh.geometry.dofmap

    # finite element dof layout
    dof_layout = V.dofmap.dof_layout

    # local owned finite element dof -> original M input id
    dof_input_ids = np.full(num_owned, -1, dtype=np.int64)

    for c in range(num_cells):
        cell_dofs = np.asarray(V.dofmap.cell_dofs(c), dtype=np.int64)
        geom_dofs = _as_row(geom_dofmap, c)

        # Number of vertices. For CG1/Q1, only vertex dofs are needed.
        vertices = np.asarray(c_to_v.links(c), dtype=np.int64)
        nverts = len(vertices)

        if len(geom_dofs) < nverts:
            raise RuntimeError(
                f"{label}: cell {c} 的 geometry dofs 数量小于顶点数，"
                f"len(geom_dofs)={len(geom_dofs)}, nverts={nverts}"
            )

        for local_vertex in range(nverts):
            # CG1/Q1: each vertex should have exactly one dof
            try:
                entity_dofs = np.asarray(
                    dof_layout.entity_dofs(0, local_vertex), dtype=np.int64
                )
            except Exception:
                # Defensive fallback: in most CG1 spaces, the first nverts entries of cell_dofs are vertex dofs
                entity_dofs = np.asarray([local_vertex], dtype=np.int64)

            if len(entity_dofs) != 1:
                raise RuntimeError(
                    f"{label}: 顶点 {local_vertex} 上不是 1 个 dof，"
                    f"entity_dofs={entity_dofs}。"
                    "当前函数只适用于标量 CG1/Q1 空间。"
                )

            cell_local_dof = int(entity_dofs[0])
            if cell_local_dof >= len(cell_dofs):
                raise RuntimeError(
                    f"{label}: cell_local_dof={cell_local_dof} 超过 "
                    f"cell_dofs 长度 {len(cell_dofs)}"
                )

            local_dof = int(cell_dofs[cell_local_dof])

            # Record only owned dofs; ghost dofs are not used as sources
            if 0 <= local_dof < num_owned:
                geom_lid = int(geom_dofs[local_vertex])
                original_id = int(input_global_indices[geom_lid])

                old = dof_input_ids[local_dof]
                if old == -1:
                    dof_input_ids[local_dof] = original_id
                elif old != original_id:
                    raise RuntimeError(
                        f"{label}: local dof {local_dof} 被映射到两个不同的 "
                        f"original input id: {old} 和 {original_id}"
                    )

    missing = np.where(dof_input_ids < 0)[0]
    if len(missing) > 0:
        raise RuntimeError(
            f"{label}: 有 {len(missing)} 个 owned dof 没有找到 original input id。"
            f"前 20 个 missing local dofs = {missing[:20]}"
        )

    return dof_input_ids

# ------------------------------------------------------------
# 1. Every Qn/source rank builds its own Vn owned dof -> M input id mapping
# ------------------------------------------------------------

src_input_ids_n_owned = owned_cg1_dof_input_ids(
    msh_n, Vn, num_owned_n, label=f"Qn/Vn on world rank {rank_n}"
)

# ------------------------------------------------------------
# 2. Solver ranks build their own Vm owned dof -> M input id mappings
# ------------------------------------------------------------

if is_solver_rank:
    target_input_ids_m_owned = owned_cg1_dof_input_ids(
        msh_m, Vm, num_owned_m, label=f"Qm/Vm on world rank {rank_n}"
    )

    map_src_rank = np.full(num_owned_m, -1, dtype=np.int64)
    map_src_local_dof = np.full(num_owned_m, -1, dtype=np.int64)
    map_src_global_dof = np.full(num_owned_m, -1, dtype=np.int64)
    del Vm, msh_m
    gc.collect()

# ============================================================
# 3. Build the Vm -> Vn mapping
#
# For each solver rank s: 
#   s broadcasts its Vm original ids to all Qn/source ranks; 
#   each source rank uses integer-set intersection to determine which ids it owns; 
#   the source rank sends:
#       Vm target local dof 
#       Vn source local dof 
#       Vn source global dof 
#   back to solver rank s.
# ============================================================

TAG_MAP_BASE = 24000

for s in solver_ranks:
    tag = TAG_MAP_BASE + s

    # --------------------------------------------------------
    # 3.1 Broadcast the Vm original ids of solver rank s
    # --------------------------------------------------------
    if rank_n == s:
        n_target = np.array([len(target_input_ids_m_owned)], dtype=np.int64)
    else:
        n_target = np.empty(1, dtype=np.int64)

    comm_n.Bcast(n_target, root=s)
    n_target = int(n_target[0])

    if rank_n == s:
        target_ids_s = np.ascontiguousarray(target_input_ids_m_owned, dtype=np.int64)
    else:
        target_ids_s = np.empty(n_target, dtype=np.int64)

    comm_n.Bcast(target_ids_s, root=s)

    # --------------------------------------------------------
    # 3.2 The current source rank determines which original ids requested by solver rank s it owns
    # --------------------------------------------------------
    #
    # common_ids[k] = M input id present in both src_input_ids_n_owned and target_ids_s
    # idx_src[k]    = position of this id in the current source rank's Vn owned-dof array
    # idx_tgt[k]    = position of this id in solver rank s's Vm owned-dof array
    #
    common_ids, idx_src, idx_tgt = np.intersect1d(
        src_input_ids_n_owned,
        target_ids_s,
        assume_unique=True,
        return_indices=True,
    )

    idx_src = idx_src.astype(np.int64, copy=False)
    idx_tgt = idx_tgt.astype(np.int64, copy=False)

    send_count = int(len(idx_tgt))

    if send_count > 0:
        send_data = np.column_stack(
            (
                idx_tgt,
                local_dofs_n_owned[idx_src].astype(np.int64, copy=False),
                global_dofs_n_owned[idx_src].astype(np.int64, copy=False),
            )
        )
        send_data = np.ascontiguousarray(send_data, dtype=np.int64)
    else:
        send_data = np.empty((0, 3), dtype=np.int64)

    # --------------------------------------------------------
    # 3.3 First gather each source rank's return count at solver rank s
    # --------------------------------------------------------
    send_count_arr = np.array([send_count], dtype=np.int64)

    if rank_n == s:
        recv_counts = np.empty(size_n, dtype=np.int64)
    else:
        recv_counts = None

    comm_n.Gather(send_count_arr, recv_counts, root=s)

    # --------------------------------------------------------
    # 3.4 Source ranks send mapping information back to solver rank s
    # --------------------------------------------------------
    if rank_n == s:
        # First handle the part where the solver rank is itself a source rank
        if send_count > 0:
            target_lids = send_data[:, 0]
            map_src_rank[target_lids] = rank_n
            map_src_local_dof[target_lids] = send_data[:, 1]
            map_src_global_dof[target_lids] = send_data[:, 2]

        # Then receive contributions from the other source ranks
        for r in range(size_n):
            if r == s:
                continue

            count_r = int(recv_counts[r])
            if count_r == 0:
                continue

            recv_data = np.empty((count_r, 3), dtype=np.int64)
            comm_n.Recv(recv_data, source=r, tag=tag)

            target_lids = recv_data[:, 0]
            map_src_rank[target_lids] = r
            map_src_local_dof[target_lids] = recv_data[:, 1]
            map_src_global_dof[target_lids] = recv_data[:, 2]

    else:
        if send_count > 0:
            comm_n.Send(send_data, dest=s, tag=tag)


# ------------------------------------------------------------
# 4. Check that every Vm owned dof on each solver rank has a corresponding Qn/Vn owned dof
# ------------------------------------------------------------

if is_solver_rank:
    missing = np.where(map_src_rank < 0)[0]
    if len(missing) > 0:
        msg = []
        msg.append(f"[world rank {rank_n}] Qm -> Qn 映射失败")
        msg.append(f"  missing number = {len(missing)}")
        msg.append(f"  first missing Vm local dofs = {missing[:20]}")
        msg.append(
            f"  first missing original ids = "
            f"{target_input_ids_m_owned[missing[:20]]}"
        )
        raise RuntimeError("\n".join(msg))


if is_solver_rank:
    request_lids_by_src_rank = {}
    target_lids_by_src_rank = {}

    for r in range(size_n):
        target_lids = np.where(map_src_rank == r)[0].astype(np.int64)
        source_lids = map_src_local_dof[target_lids].astype(np.int64)

        request_lids_by_src_rank[r] = source_lids
        target_lids_by_src_rank[r] = target_lids

    active_sources_to_this_solver = [
        r for r in range(size_n)
        if len(target_lids_by_src_rank[r]) > 0
    ]

# ============================================================
# 5. Each solver rank creates its own request table
#    The subsequent communication-initialization code can retain the original logic
# ============================================================

# ============================================================
# Distribute request tables using numeric buffers and Scatterv
# Avoid pickle, MPI_Mprobe, and a large number of incomplete isend operations
# ============================================================

req_from_solver = {}

mpi_int64 = (
    MPI.INT64_T
    if hasattr(MPI, "INT64_T")
    else MPI.LONG_LONG
)

for s in solver_ranks:

    # --------------------------------------------------------
    # 1. Solver rank s computes the array length sent to each world rank
    # --------------------------------------------------------
    if rank_n == s:
        counts = np.asarray(
            [
                len(request_lids_by_src_rank[r])
                for r in range(size_n)
            ],
            dtype=np.int32,
        )
    else:
        counts = None

    # Each rank first receives its own array length
    recv_count = np.empty(1, dtype=np.int32)

    comm_n.Scatter(
        counts,
        recv_count,
        root=s,
    )

    local_count = int(recv_count[0])
    recv_data = np.empty(local_count, dtype=np.int64)

    # --------------------------------------------------------
    # 2. The root packs request arrays for all target ranks
    # --------------------------------------------------------
    if rank_n == s:
        displs = np.zeros(size_n, dtype=np.int32)

        if size_n > 1:
            np.cumsum(
                counts[:-1],
                out=displs[1:],
            )

        total_count = int(np.sum(counts, dtype=np.int64))

        if total_count > 0:
            packed_data = np.concatenate(
                [
                    np.asarray(
                        request_lids_by_src_rank[r],
                        dtype=np.int64,
                    )
                    for r in range(size_n)
                ]
            )
            packed_data = np.ascontiguousarray(
                packed_data,
                dtype=np.int64,
            )
        else:
            packed_data = np.empty(0, dtype=np.int64)

        scatter_sendbuf = [
            packed_data,
            counts,
            displs,
            mpi_int64,
        ]
    else:
        scatter_sendbuf = None

    # --------------------------------------------------------
    # 3. Distribute variable-length int64 request arrays
    # --------------------------------------------------------
    comm_n.Scatterv(
        scatter_sendbuf,
        recv_data,
        root=s,
    )

    req_from_solver[s] = recv_data


if is_solver_rank:
    request_lids_by_src_rank = {}
    target_lids_by_src_rank = {}

    for r in range(size_n):
        target_lids = np.where(map_src_rank == r)[0].astype(np.int64)
        source_lids = map_src_local_dof[target_lids].astype(np.int64)

        request_lids_by_src_rank[r] = source_lids
        target_lids_by_src_rank[r] = target_lids

    active_sources_to_this_solver = [
        r for r in range(size_n)
        if len(target_lids_by_src_rank[r]) > 0
    ]


active_solvers_from_this_source = [
    s for s in solver_ranks
    if len(req_from_solver[s]) > 0
]


# ============================================================
# 8. Preallocate communication buffers
# ============================================================

# n -> m: send buffers from source ranks to solver ranks
sendbuf_n_to_m = {
    s: np.empty(len(req_from_solver[s]), dtype=value_dtype)
    for s in active_solvers_from_this_source
}

# m -> n: receive buffers for source ranks to receive values from solver ranks
recvbuf_m_to_n = {
    s: np.empty(len(req_from_solver[s]), dtype=value_dtype)
    for s in active_solvers_from_this_source
}

if is_solver_rank:
    # n -> m: receive buffers for a solver rank to receive from each source rank
    recvbuf_n_to_m = {
        r: np.empty(len(target_lids_by_src_rank[r]), dtype=value_dtype)
        for r in active_sources_to_this_solver
    }

    # m -> n: send buffers for a solver rank to send back to each source rank
    sendbuf_m_to_n = {
        r: np.empty(len(target_lids_by_src_rank[r]), dtype=value_dtype)
        for r in active_sources_to_this_solver
    }


local_num_values = np.array(num_owned_n, dtype=np.int64)
global_num_values = comm_n.allreduce(local_num_values, op=MPI.SUM)

local_num_edges = np.array(len(active_solvers_from_this_source), dtype=np.int64)
global_num_edges = comm_n.allreduce(local_num_edges, op=MPI.SUM)
max_edges_per_source = comm_n.allreduce(local_num_edges, op=MPI.MAX)

if is_solver_rank:
    local_solver_sources = np.array(len(active_sources_to_this_solver), dtype=np.int64)
else:
    local_solver_sources = np.array(0, dtype=np.int64)

max_sources_per_solver = comm_n.allreduce(local_solver_sources, op=MPI.MAX)

if rank_n == 0:
    print("")
    print("============================================================")
    print("Communication setup summary")
    print("============================================================")
    print(f"Total ranks n = {size_n}")
    print(f"Solver ranks m = {m}")
    print(f"Total Vn owned values = {global_num_values}")
    print(f"Nonempty communication pairs = {global_num_edges}")
    print(f"Max active solver targets per source rank = {max_edges_per_source}")
    print(f"Max active source ranks per solver rank = {max_sources_per_solver}")
    print("============================================================")
    print("")

TAG_VAL_BASE = 10000
TAG_BACK_BASE = 20000


def create_owned_petsc_vec(num_local, comm):
    """创建一个本地 owned size 精确等于 num_local 的 PETSc Vec。"""
    v = PETSc.Vec().createMPI((num_local, PETSc.DECIDE), comm=comm)
    v.set(0.0)

    return v

def vec_n_to_m(
    vec_n: PETSc.Vec,
    out_vec_m: PETSc.Vec | None = None,
    check: bool = True,
):
    """
    把定义在 n 个进程上的 PETSc Vec 通信到前 m 个 solver ranks。

    Parameters
    ----------
    vec_n
        n 个进程上都有的源 PETSc Vec。

    out_vec_m
        可选。solver ranks 上的输出 PETSc Vec。
        若为 None，则 solver ranks 内部创建一个 Vm-owned size 的 PETSc Vec；
        非 solver ranks 会忽略这个参数并返回 None。

    check
        是否检查 m 进程上的 owned dofs 是否全部被填入。

    Returns
    -------
    PETSc.Vec | None
        solver ranks 返回 out_vec_m；非 solver ranks 返回 None。
    """

    # ------------------------------------------------------------
    # 0. Create the output Vec on the m processes
    # ------------------------------------------------------------
    if is_solver_rank:
        if out_vec_m is None:
            out_vec_m = create_owned_petsc_vec(num_owned_m, solve_comm)

        if check:
            if out_vec_m.getLocalSize() != num_owned_m:
                raise RuntimeError(
                    f"[rank {rank_n}] out_vec_m local size mismatch: "
                    f"expected {num_owned_m}, got {out_vec_m.getLocalSize()}"
                )

    else:
        out_vec_m = None

    if check:
        if vec_n.getLocalSize() != num_owned_n:
            raise RuntimeError(
                f"[rank {rank_n}] vec_n local size mismatch: "
                f"expected {num_owned_n}, got {vec_n.getLocalSize()}"
            )

    # ------------------------------------------------------------
    # 1. Solver ranks pre-post Irecv operations
    # ------------------------------------------------------------
    recv_reqs = []
    recv_sources = []

    if is_solver_rank:
        for r in active_sources_to_this_solver:
            if r == rank_n:
                continue

            recv_reqs.append(
                comm_n.Irecv(
                    recvbuf_n_to_m[r],
                    source=r,
                    tag=TAG_VAL_BASE + rank_n,
                )
            )
            recv_sources.append(r)

    # ------------------------------------------------------------
    # 2. All n ranks send their owned values to the corresponding solver ranks
    # ------------------------------------------------------------
    send_reqs = []

    # Standard PETSc MPI Vec: read the local owned array directly
    x_n_owned = vec_n.array_r

    if is_solver_rank:
        # Standard PETSc MPI Vec: write the local owned array directly
        x_m_owned = out_vec_m.array_w

        filled_m = np.zeros(num_owned_m, dtype=bool) if check else None

        for s in active_solvers_from_this_source:
            source_lids = req_from_solver[s]

            if rank_n == s:
                # Self-transfer; do not use MPI
                target_lids = target_lids_by_src_rank[rank_n]

                x_m_owned[target_lids] = x_n_owned[source_lids]

                if check:
                    filled_m[target_lids] = True

            else:
                # Copy into the send buffer
                sendbuf_n_to_m[s][:] = x_n_owned[source_lids]

                send_reqs.append(
                    comm_n.Isend(
                        sendbuf_n_to_m[s],
                        dest=s,
                        tag=TAG_VAL_BASE + s,
                    )
                )

        # ------------------------------------------------------------
        # 3. Wait for receives to complete, then write into out_vec_m
        # ------------------------------------------------------------
        if len(recv_reqs) > 0:
            MPI.Request.Waitall(recv_reqs)

        for r in recv_sources:
            target_lids = target_lids_by_src_rank[r]

            x_m_owned[target_lids] = recvbuf_n_to_m[r]

            if check:
                filled_m[target_lids] = True

        if check:
            missing_m = np.count_nonzero(~filled_m)
            if missing_m > 0:
                missing_lids = np.where(~filled_m)[0]
                raise RuntimeError(
                    f"[world rank {rank_n}] n->m 后 Vm 有 {missing_m} 个 dof 没有填入。"
                    f" missing_lids = {missing_lids[:50]}"
                )

        # Optional, but recommended
        out_vec_m.assemble()

    else:
        # Non-solver ranks only need to send to solver ranks
        for s in active_solvers_from_this_source:
            source_lids = req_from_solver[s]

            sendbuf_n_to_m[s][:] = x_n_owned[source_lids]

            send_reqs.append(
                comm_n.Isend(
                    sendbuf_n_to_m[s],
                    dest=s,
                    tag=TAG_VAL_BASE + s,
                )
            )

    # ------------------------------------------------------------
    # 4. Ensure the send buffers can be safely reused
    # ------------------------------------------------------------
    if len(send_reqs) > 0:
        MPI.Request.Waitall(send_reqs)

    return out_vec_m

def vec_m_to_n(
    vec_m: PETSc.Vec | None,
    out_vec_n: PETSc.Vec | None = None,
    check: bool = True,
):
    """
    把前 m 个 solver ranks 上的 PETSc Vec 通信回 n 个进程。

    Parameters
    ----------
    vec_m
        solver ranks 上的源 PETSc Vec。
        非 solver ranks 可以传 None。
        这里假设 vec_m 是普通 MPI Vec，直接读取 vec_m.array_r。

    out_vec_n
        可选。n 个进程上的输出 PETSc Vec。
        若为 None，则所有 n ranks 内部创建一个 Vn-owned size 的 PETSc Vec。

    check
        是否检查 n 进程上的 owned dofs 是否全部被填入。

    Returns
    -------
    PETSc.Vec
        n 个进程上的输出 PETSc Vec。
    """

    # ------------------------------------------------------------
    # 0. Create the output Vec on the n processes
    # ------------------------------------------------------------
    if out_vec_n is None:
        out_vec_n = create_owned_petsc_vec(num_owned_n, comm_n)

    if is_solver_rank and vec_m is None:
        raise RuntimeError(
            f"[world rank {rank_n}] solver rank 调用 vec_m_to_n 时 vec_m 不能为 None。"
        )

    if check:
        if out_vec_n.getLocalSize() != num_owned_n:
            raise RuntimeError(
                f"[rank {rank_n}] out_vec_n local size mismatch: "
                f"expected {num_owned_n}, got {out_vec_n.getLocalSize()}"
            )

        if is_solver_rank:
            if vec_m.getLocalSize() != num_owned_m:
                raise RuntimeError(
                    f"[rank {rank_n}] vec_m local size mismatch: "
                    f"expected {num_owned_m}, got {vec_m.getLocalSize()}"
                )

    # ------------------------------------------------------------
    # 1. All n ranks first pre-post Irecv operations
    #    Prepare to receive returned values from the relevant solver ranks
    # ------------------------------------------------------------
    recv_reqs = []
    recv_solvers = []

    for s in active_solvers_from_this_source:
        if s == rank_n:
            continue

        recv_reqs.append(
            comm_n.Irecv(
                recvbuf_m_to_n[s],
                source=s,
                tag=TAG_BACK_BASE + rank_n,
            )
        )
        recv_solvers.append(s)

    send_reqs = []

    # ------------------------------------------------------------
    # 2. Write directly into out_vec_n's local owned array
    # ------------------------------------------------------------
    x_n_out = out_vec_n.array_w
    filled_n = np.zeros(num_owned_n, dtype=bool) if check else None

    # ------------------------------------------------------------
    # 3. Solver ranks send data from vec_m back to the corresponding n ranks
    # ------------------------------------------------------------
    if is_solver_rank:
        # Standard PETSc MPI Vec: read the local owned array directly
        x_m_owned = vec_m.array_r

        for r in active_sources_to_this_solver:
            target_lids = target_lids_by_src_rank[r]

            if r == rank_n:
                # Self-transfer; do not use MPI
                source_lids = req_from_solver[rank_n]

                x_n_out[source_lids] = x_m_owned[target_lids]

                if check:
                    filled_n[source_lids] = True

            else:
                sendbuf_m_to_n[r][:] = x_m_owned[target_lids]

                send_reqs.append(
                    comm_n.Isend(
                        sendbuf_m_to_n[r],
                        dest=r,
                        tag=TAG_BACK_BASE + r,
                    )
                )

    # ------------------------------------------------------------
    # 4. Wait for receives to complete, then write into out_vec_n
    # ------------------------------------------------------------
    if len(recv_reqs) > 0:
        MPI.Request.Waitall(recv_reqs)

    for s in recv_solvers:
        source_lids = req_from_solver[s]

        x_n_out[source_lids] = recvbuf_m_to_n[s]

        if check:
            filled_n[source_lids] = True

    if check:
        missing_n = np.count_nonzero(~filled_n)

        if missing_n > 0:
            missing_lids = np.where(~filled_n)[0]
            raise RuntimeError(
                f"[world rank {rank_n}] m->n 后 Vn 有 {missing_n} 个 dof 没有填入。"
                f" missing_lids = {missing_lids[:50]}"
            )

    # The PETSc Vec has been modified; assemble after local writes
    out_vec_n.assemble()

    # ------------------------------------------------------------
    # 5. Ensure the send buffers can be safely reused
    # ------------------------------------------------------------
    if len(send_reqs) > 0:
        MPI.Request.Waitall(send_reqs)
    return out_vec_n

# ------------------------------------------------------------------------------
delta_t = fem.Constant(msh, default_real_type(t_end / num_time_steps)) 
alpha = fem.Constant(msh, default_real_type(6.0)) 

h = CellDiameter(msh)
n = FacetNormal(msh)
n_vec = FacetNormal(msh)
x = ufl.SpatialCoordinate(msh)

def mean(vn,vn_1): 
    return 1/2*(vn+vn_1) 

def star(v_n_1,v_n_2): 
    return 1/2*(3*v_n_1-v_n_2) 

def jump(phi, n): 
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-")) 
def conv_u(w, z, v):
    return 0.5 * (
          inner(dot(grad(z), w), v)
        - inner(dot(grad(v), w), z)
    ) * dx 

def conv_T(w, theta, Tv):
    return 0.5 * (
          inner(dot(w, grad(theta)), Tv)
        - inner(dot(w, grad(Tv)), theta)
    ) * dx 

NT_n   = conv_T(u_n_1, T_n_1, Tv)
NT_nm1 = conv_T(u_n_2, T_n_2, Tv)

conv_TT = 1.5 * NT_n - 0.5 * NT_nm1

Nu_n   = conv_u(u_n_1, u_n_1, v)
Nu_nm1 = conv_u(u_n_2, u_n_2, v)

conv_uu = 1.5 * Nu_n - 0.5 * Nu_nm1

###Upwind-scheme operator 
lmbda = conditional(gt(dot(star(u_n_1,u_n_2), n), 0), 1, 0)  
u_uw = lmbda("+") * mean(u,u_n_1)("+") + lmbda("-") * mean(u,u_n_1)("-")  

###  Navier-Stokes equations 
###  Time-derivative term 
a_00=Ekman*(inner(u / delta_t, v) * dx - inner(u_n_1 / delta_t, v) * dx) 

##Weak boundary term for convection   

a_00+=Ekman*conv_uu
##Coriolis force   
a_00+=2*inner(cross(ufl.as_vector([0,0,1]),mean(u,u_n_1)),v)*dx  

a_00+=Ekman*inner(div(u),div(v))*dx  

a_00+=Ekman*(inner(grad(mean(u,u_n_1)), grad(v)) * dx)  
## Numerical stabilization term 
#F += 2*Ekman/delta_t *inner(div(u),div(v)) *dx  
zeros=fem.Constant(msh, default_real_type(0))  
# ## Pressure term  
a_01= -inner(P, div(v)) * dx     
a_10= -inner(div(u), q) * dx     
a_11= inner(zeros*P,q)*dx       
## Lorentz force  
a_02  = -1.0/Pm * inner(cross(curl(mean(A,A_n_1)),star(A_n_1,A_n_2)),v)*dx  
## Buoyancy term  
a_03 = -Ra/ro * inner(ufl.as_vector([x[0],x[1],x[2]])*star(T_n_1,T_n_2),v) *dx     
a_03 +=Ra/ro * inner(ufl.as_vector([x[0],x[1],x[2]])*(ri*ro/norm_r(x)-ri),v) *dx   

a_22 = inner(A/ delta_t,Av) * dx -inner(A_n_1/ delta_t,Av) * dx    
a_20 = inner(cross(curl(Av),star(A_n_1,A_n_2)),mean(u,u_n_1))*dx   
a_22 += 1/Pm *inner(curl(mean(A,A_n_1)),curl(Av)) *dx              
a_22 += inner(div(A),div(Av)) *dx  


a_33 =inner(T / delta_t, Tv) * dx -inner(T_n_1 / delta_t, Tv) * dx   
a_33 +=1/Pr*inner(grad(mean(T,T_n_1)),grad(Tv))*dx   
a_33+=conv_TT

a_00_final=fem.form(ufl.lhs(a_00))
a_01_final=fem.form(ufl.lhs(a_01))
a_02_final=fem.form(ufl.lhs(a_02))
a_20_final=fem.form(ufl.lhs(a_20))


zero_dynamic = fem.Constant(msh, default_scalar_type(0.0))
a_02_zero_final = fem.form(zero_dynamic * inner(A, v) * dx)
a_20_zero_final = fem.form(zero_dynamic * inner(u, Av) * dx)
a_33_zero_final = fem.form(zero_dynamic * inner(T, Tv) * dx)


a_11_final=fem.form(a_11) 
a_10_final=fem.form(ufl.lhs(a_10)) 

a_22_final=fem.form(ufl.lhs(a_22)) 

a_33_final=fem.form(ufl.lhs(a_33)) 

L0=fem.form(ufl.rhs(a_00)+ufl.rhs(a_01)+ufl.rhs(a_02)+ufl.rhs(a_03))

L1 = fem.form(inner(fem.Constant(msh,0.0), q) * dx)
L2=fem.form(ufl.rhs(a_22)+ufl.rhs(a_20))
L3=fem.form(ufl.rhs(a_33))

a = [    
        [a_00_final, a_01_final, a_02_final,     None,    ],
        [a_10_final, a_11_final,     None,       None,    ],
        [a_20_final,     None,   a_22_final,     None,    ],
        [   None,        None,       None,      a_33_final,],
    ]    

# Static matrix: zero-valued placeholders for dynamic coupling blocks preserve the sparsity pattern.
a_static = [
        [a_00_final, a_01_final,   a_02_zero_final, None,    ],
        [a_10_final, a_11_final,        None,      None,    ],
        [a_20_zero_final, None,       a_22_final,  None,    ],
        [   None,        None,            None,  a_33_final,],
    ]

# Sparsity pattern of the dynamic-coupling auxiliary matrix.
# Blocks (0, 2) and (2, 0) correspond to A02 and A20; the zero diagonal blocks for pressure and temperature
# only preserve the complete 4 x 4 block layout and do not participate in the actual dynamic-coupling calculation.
a_dynamic_02_pattern = [
        [   None,          None,    a_02_zero_final,       None,],
        [   None,    a_11_final,               None,       None,],
        [a_20_zero_final,  None,               None,       None,],
        [   None,          None,               None, a_33_zero_final,],
    ]

L = [L0, L1, L2, L3]  

ap=inner(grad(P), grad(q)) * dx  
ap+=0.1/h*inner(P,q)*ds  

fp=2*Ekman*(1/delta_t)*inner(P,q)*dx   
fp+=Ekman*inner(grad(P),grad(q))*dx   
#fp+=0.1/h*inner(P,q)*ds   # 0.1 is important; 0.01 causes severe convergence degradation
#fp += Ekman*0.5*inner(dot(star(u_n_1,u_n_2),grad(P)),q)*dx   
#fp -= Ekman*0.5*inner(dot(star(u_n_1,u_n_2),grad(q)),P)*dx   

Mp=fem.form(inner(P,q)*dx)    
Ap=fem.form(ap)     
Fp=fem.form(fp)     

a_p = ([   
        [a_00_final, a_01_final, a_02_final,      None,   ],
        [   None,    a_11_final,   None,          None,   ],
        [   None,     None,      a_22_final,      None,   ],
        [   None,     None,       None,        a_33_final,  ],
    ])  

P=fem.petsc.create_matrix_nest(a_p)   
P.assemble()   
nested_IS = P.getNestISs()    
P.destroy()   


# Assemble the static stiffness matrix only once.
A_static = create_matrix_block(a_static)
A_static.zeroEntries()
assemble_matrix_block(A_static, a_static, bcs=bcs)  # type: ignore
A_static.assemble()

A = A_static.copy()
b = create_vector_block(L)
x = A.createVecRight()

_block_spaces = [U_space, P_space, A_space, T_space]
_block_is = dolfinx.cpp.la.petsc.create_index_sets([
    (V.dofmap.index_map, V.dofmap.index_map_bs) for V in _block_spaces
])
_bcs_cpp = [bc._cpp_object for bc in bcs]


# Create the dynamic-coupling auxiliary matrix only once. First activate its sparsity pattern with zero forms, then zero it;
# Within each time step, assemble a_02_final only in block (0, 2).
A02_aux = create_matrix_block(a_dynamic_02_pattern)
A02_aux.zeroEntries()
assemble_matrix_block(A02_aux, a_dynamic_02_pattern, bcs=[])  # type: ignore
A02_aux.assemble()
A02_aux.zeroEntries()
A02_aux.assemble()

# A02_aux_T always stores the explicit transpose of A02_aux. The matrix and its sparsity pattern are reused at every
# time step to avoid repeatedly creating PETSc matrices.
A02_aux_T = A02_aux.copy()
A02_aux_T.transpose()

# PETSc >= 3.18 requires the matrix passed to MatTranspose(MAT_REUSE_MATRIX) to
# come from a MAT_INITIAL_MATRIX transpose or be explicitly registered as a transpose precursor.
# Here A02_aux_T is obtained by an in-place transpose after copy(), so this relationship must be registered.
_reuse_A02_transpose = hasattr(A02_aux, "setTransposePrecursor")
if _reuse_A02_transpose:
    A02_aux.setTransposePrecursor(A02_aux_T)


P0 = A.createSubMatrix(nested_IS[0][0], nested_IS[0][0])
P3 = A.createSubMatrix(nested_IS[0][3], nested_IS[0][3])
P02 = A02_aux.createSubMatrix(nested_IS[0][0], nested_IS[0][2])
P0.assemble()
P3.assemble()
P02.assemble()

# The pressure-gradient block is time-independent; assemble it once separately.
P01 = assemble_matrix(a_01_final, bcs=bcs)
P01.assemble()

P0 = assemble_matrix(a_00_final, bcs=bcs)  
P0.assemble() 
b0 = create_vector(L0)
x0 = b0.duplicate()
ksp_g0 = PETSc.KSP().create(msh.comm)  # type: ignore
ksp_g0.setOperators(P0)
ksp_g0.setType('gmres')
ksp_g0.getPC().setType(PETSc.PC.Type.ASM)
opts = PETSc.Options()  # type: ignore
opts["ksp_rtol"] = 1.0e-3     # Residual tolerance
opts["ksp_atol"] = 1.0e-6     # Absolute residual tolerance
opts["ksp_max_it"] = 2000
ksp_g0.setFromOptions()
#ksp_g0.setMonitor(monitor_residual)

# Keep only the pressure-block vector on the full communicator; no longer create P1/ksp_g1/nullspace.
b1 = create_vector(L1)
x1 = b1.duplicate()

P12 = assemble_matrix(Mp, bcs=[])
P12.assemble()
ksp_g12 = PETSc.KSP().create(msh.comm)  # type: ignore
ksp_g12.setOperators(P12)
ksp_g12.setType('cg')
ksp_g12.setInitialGuessNonzero(False)
ksp_g12.getPC().setType(PETSc.PC.Type.JACOBI)
opts = PETSc.Options()  # type: ignore
opts["ksp_rtol"] = 1.0e-3
opts["ksp_atol"] = 1.0e-6
opts["ksp_max_it"] = 2000
ksp_g12.setFromOptions()

PFP = assemble_matrix(Fp, bcs=[])
PFP.assemble()

P2 = assemble_matrix(a_22_final,bcs=bcs)  
P2.assemble()  
b2 = create_vector(L2)
x2 = b2.duplicate()
P2 = assemble_matrix(a_22_final,bcs=bcs)  
P2.assemble()  
ksp_g2 = PETSc.KSP().create(msh.comm)  # type: ignore  
ksp_g2.setOperators(P2)    
ksp_g2.setType('gmres')  
ksp_g2.getPC().setType(PETSc.PC.Type.ASM)   
opts = PETSc.Options()        # type: ignore 
opts["ksp_rtol"] = 1.0e-3     ##Residual tolerance  
opts["ksp_atol"] = 1.0e-6     # Absolute residual tolerance  
opts["ksp_max_it"] =2000       # Set the maximum number of iterations to 800  
#opts["-pc_asm_blocks "] =64 
ksp_g2.setFromOptions()   
#ksp_g2.setMonitor(monitor_residual)   

P3 = assemble_matrix(a_33_final, bcs=bcs)
P3.assemble()
b3 = create_vector(L3)
x3 = b3.duplicate()
ksp_g3 = PETSc.KSP().create(msh.comm)  # type: ignore
ksp_g3.setOperators(P3)
ksp_g3.setType('gmres')
ksp_g3.getPC().setType(PETSc.PC.Type.ASM)
opts = PETSc.Options()  # type: ignore
opts["ksp_rtol"] = 1.0e-3
opts["ksp_atol"] = 1.0e-6
opts["ksp_max_it"] = 2000
ksp_g3.setFromOptions()
#ksp_g3.setMonitor(monitor_residual)

x_global=b.copy()   
revmode = PETSc.Scatter.Mode.REVERSE    
sct0 = PETSc.Scatter().create(x_global,nested_IS[0][0],x0,None)   
sct1 = PETSc.Scatter().create(x_global,nested_IS[0][1],x1,None)   
sct2 = PETSc.Scatter().create(x_global,nested_IS[0][2],x2,None)   
sct3 = PETSc.Scatter().create(x_global,nested_IS[0][3],x3,None)   

sct_b0 = PETSc.Scatter().create(b,nested_IS[0][0],b0,None)  
sct_b1 = PETSc.Scatter().create(b,nested_IS[0][1],b1,None)  
sct_b2 = PETSc.Scatter().create(b,nested_IS[0][2],b2,None)  
sct_b3 = PETSc.Scatter().create(b,nested_IS[0][3],b3,None) 

# Work vectors used by every Python preconditioner call; allocate them only once at startup.
b_tmp0 = b0.duplicate()
b_tmp1 = b1.duplicate()

comm = MPI.COMM_WORLD    
rank=MPI.COMM_WORLD.Get_rank()    
size = comm.Get_size()    
 
t=0   
u_vis = fem.Function(WW)   
P_vis = fem.Function(TT)   
A_vis = fem.Function(WW)      
T_vis = fem.Function(TT)    

tag='plus'
u_file = io.VTXWriter(msh.comm, "u_dynamo_"+tag+".bp4", [u_vis._cpp_object],engine="BP4") 
P_file = io.VTXWriter(msh.comm, "P_dynamo_"+tag+".bp4", [P_vis._cpp_object],engine="BP4") 
A_file = io.VTXWriter(msh.comm, "A_dynamo_"+tag+".bp4", [A_vis._cpp_object],engine="BP4") 
T_file = io.VTXWriter(msh.comm, "T_dynamo_"+tag+".bp4", [T_vis._cpp_object],engine="BP4") 

u_file.write(t) 
A_file.write(t) 
T_file.write(t) 

dof_coordinates = WW.tabulate_dof_coordinates() 
num_local_dofs = WW.dofmap.index_map.size_local 

u_h=fem.Function(U_space) 
p_h=fem.Function(P_space) 
A_h=fem.Function(A_space) 
T_h=fem.Function(T_space) 

def _assemble_one_dynamic_block(target_matrix, row, col, form):
    """Assemble one finite-element block into a monolithic block matrix."""
    constants = dolfinx.cpp.fem.pack_constants(form._cpp_object)
    coeffs = dolfinx.cpp.fem.pack_coefficients(form._cpp_object)
    A_local = target_matrix.getLocalSubMatrix(
        _block_is[row],
        _block_is[col],
    )
    dolfinx.cpp.fem.petsc.assemble_matrix(
        A_local,
        form._cpp_object,
        constants,
        coeffs,
        _bcs_cpp,
        True,
    )
    target_matrix.restoreLocalSubMatrix(
        _block_is[row],
        _block_is[col],
        A_local,
    )

def assemble_dynamic_blocks_only():
    """只组装 A02，并用 A20 = -Pm * A02^T 构造另一耦合块。"""

    global A02_aux_T

    # Perform one finite-element cell integration only: assemble block (0, 2), A02.
    A02_aux.zeroEntries()
    _assemble_one_dynamic_block(
        A02_aux,
        0,
        2,
        a_02_final,
    )
    A02_aux.assemble()

    # Reuse the transpose matrix sparsity pattern and construct A20 = -Pm * A02^T.
    # If an older petsc4py lacks setTransposePrecursor, recreate at each step
    # the explicit transpose matrix to ensure compatibility and correctness.
    if _reuse_A02_transpose:
        A02_aux.transpose(A02_aux_T)
    else:
        A02_aux_T.destroy()
        A02_aux_T = A02_aux.copy()
        A02_aux_T.transpose()

    A02_aux_T.scale(-Pm)

    # Restore the static part, then add the two dynamic coupling blocks.
    A_static.copy(A, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
    A.axpy(
        1.0,
        A02_aux,
        structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
    )
    A.axpy(
        1.0,
        A02_aux_T,
        structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
    )
    A.assemble()
   

def matricsx_assemble():

    assemble_dynamic_blocks_only()
    
    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)  # type: ignore

    # Only P02 must be recomputed in the preconditioner
    P02.zeroEntries()
    A02_aux.createSubMatrix(nested_IS[0][0], nested_IS[0][2], P02)
    P02.assemble()

b_tmp0=b0.duplicate() # dim is dofs of u  
b_tmp1=b1.duplicate() # dim is dofs of p  
    
###Solver settings
class pcd:
    num=0
    def __init__(self):
        
        self.num = 0
    def apply(self,pc, x, x_global):

        pcd.num +=1
        time_suba=time.time()
        sct_b0.scatter(x,b0)   #bb0=bb[is1]   
        sct_b1.scatter(x,b1)   #bb0=bb[is1]   
        sct_b2.scatter(x,b2)   #bb0=bb[is1]   
        sct_b3.scatter(x,b3)   #bb0=bb[is1]   
        # Block solve 

        ksp_g3.solve(b3,x3)  
        sct3.scatter(x3,x_global,mode=revmode)   #bb0=bb[is1]  
        time1=time.time()

        ksp_g2.solve(b2,x2)  
        
        sct2.scatter(x2,x_global,mode=revmode)   #bb0=bb[is1]   

        time2=time.time()

        #vec_n_to_m(b1, out_vec_m=b_m, check=False)
        b_m = vec_n_to_m(b1)
        if is_solver_rank:
            b_m.assemble()
            ksp_gm.solve(b_m,xm)
            b_m=xm
        vec_m_to_n(b_m,x1)


        PFP.mult(x1,b_tmp1)   

        ksp_g12.solve(b_tmp1,x1)  
        

        sct1.scatter(x1,x_global,mode=revmode)   #bb0=bb[is1]   
        
        # Compute b0    
        # b0=b0-A01*x1-A02*x2       
        P01.mult(x1,b_tmp0)   #b_tmp=p01*x1    
        b0.axpy(-1,b_tmp0)   # y = -x + y   
        P02.mult(x2,b_tmp0)  
        b0.axpy(-1,b_tmp0)  
        # 
        time5=time.time()
        ksp_g0.solve(b0,x0)  
        time6=time.time()

        sct0.scatter(x0,x_global,mode=revmode)   #bb0=bb[is1]    


ksp = PETSc.KSP().create()      
ksp.setType('fgmres')      
ksp.getPC().setType(PETSc.PC.Type.PYTHON)      
ksp.getPC().setPythonContext(pcd())    
ksp.setOperators(A)    
opts = PETSc.Options()  # type: ignore      
opts["ksp_rtol"] = 1.0e-10     ##Residual tolerance       
opts["ksp_atol"] = 1.0e-8    # Absolute residual tolerance       
opts["ksp_max_it"] =2000       # Set the maximum number of iterations to 800      
opts["ksp_gmres_restart"] = 70  # Set the restart count to 70  
ksp.setFromOptions()     
ksp.setMonitor(monitor_residual)     

comm = MPI.COMM_WORLD   
rank=MPI.COMM_WORLD.Get_rank()   
size = comm.Get_size()   

if rank==0:  
    print(f'这个测试使用了个{size}个进程进行计算了连续元下的结果',flush=True) 
    print(f'检验Ekman {Ekman} 和Rm {Pm} 数 dt{delta_t.value} and PM{Pm} 对求解器的敏感性 ',flush=True) 
    print('网格是,隐式的数值格式',name,flush=True)


def checkpoint_output(t=0.0):

    checkpoint_file = f"checkpoint_big_plus.bp4"
    
    checkpoint_time = float(t)

    # Set the function names stored in the checkpoint
    u_n_1.name = "u_n_1"
    u_n_2.name = "u_n_2"

    A_n_1.name = "B_n_1"
    A_n_2.name = "B_n_2"

    T_n_1.name = "T_n_1"
    T_n_2.name = "T_n_2"
    
    # Update parallel ghost degrees of freedom
    u_n_1.x.scatter_forward()
    u_n_2.x.scatter_forward()

    A_n_1.x.scatter_forward()
    A_n_2.x.scatter_forward()

    T_n_1.x.scatter_forward()
    T_n_2.x.scatter_forward()

    # Step 1: write the mesh 
    # write_mesh uses Write mode by default and creates or overwrites the file 
    adx.write_mesh(
        checkpoint_file,
        msh,
        engine="BP4",
        time=checkpoint_time
    )

    # Step 2: append velocity to the same checkpoint file 
    adx.write_function(
        checkpoint_file,
        u_n_1,
        engine="BP4",
        time=checkpoint_time,
        name="u_n_1"
    )

    adx.write_function(
        checkpoint_file,
        u_n_2,
        engine="BP4",
        time=checkpoint_time,
        name="u_n_2"
    )
    
    # Step 3: append the magnetic field to the same checkpoint file 
    adx.write_function(
        checkpoint_file,
        A_n_1,
        engine="BP4",
        time=checkpoint_time,
        name="B_n_1"
    )

    adx.write_function(
        checkpoint_file,
        A_n_2,
        engine="BP4",
        time=checkpoint_time,
        name="B_n_2"
    )

    # Step 4: append temperature to the same checkpoint file 
    adx.write_function(
        checkpoint_file,
        T_n_1,
        engine="BP4",
        time=checkpoint_time,
        name="T_n_1"
    )

    adx.write_function(
        checkpoint_file,
        T_n_2,
        engine="BP4",
        time=checkpoint_time,
        name="T_n_2"
    )

    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0: 
        print("Checkpoint 输出完成") 
        print(f"文件名：{checkpoint_file}") 
        print(f"保存时间：{checkpoint_time}")  
        print("包含变量:u, B, T") 

for n in range(10000*5*10):
    
    t += delta_t.value
    
    time_begin=time.time()
    matricsx_assemble()
    time_assembel=time.time()
    if rank==0:
        print('组装用时,',time_assembel-time_begin,flush=True)
    
    ksp.solve(b,x)
    time_end=time.time()
    if rank==0:
        print('1个时间步求解用时:,',time_end-time_assembel,flush=True)
    
    u_h.x.array[:u_offset] = x.array_r[:u_offset] 
    p_h.x.array[:p_offset-u_offset]=x.array_r[u_offset:p_offset] 
    A_h.x.array[:A_offset-p_offset]=x.array_r[p_offset:A_offset] 
    T_h.x.array[:T_offset-A_offset]=x.array_r[A_offset:T_offset] 
    
    p_h.x.array[:] -= domain_average(msh, p_h)
    
    u_h.x.scatter_forward() 
    p_h.x.scatter_forward() 
    A_h.x.scatter_forward() 
    T_h.x.scatter_forward() 
    
    if n % 500==0: 
        ekin=norm_L2(msh.comm,u_h)
        mkin=norm_L2(msh.comm,A_h)

        if MPI.COMM_WORLD.Get_rank() ==0: 
            with open("Ekin_"+tag+".txt","a") as ekinfile:     
                ekinfile.write(str(t)+'\t'+str(ekin)+'\n') 
            ekinfile.close() 

            with open("Mkin_"+tag+".txt","a") as mkinfile:     
                mkinfile.write(str(t)+'\t'+str(mkin)+'\n') 
            mkinfile.close()
    
    if n % 5000==0: 
        #Output a BP file for plotting 
        u_vis.interpolate(u_h)  
        P_vis.interpolate(p_h)  
        A_vis.interpolate(A_h)  
        T_vis.interpolate(T_h)  

        u_file.write(t) 
        A_file.write(t) 
        T_file.write(t) 
        P_file.write(t) 

        checkpoint_output(t)

    # Update u_n
    u_n_2.x.array[:] = u_n_1.x.array[:]
    u_n_1.x.array[:] = u_h.x.array[:]

    A_n_2.x.array[:] = A_n_1.x.array[:]
    A_n_1.x.array[:] = A_h.x.array[:]

    T_n_2.x.array[:] = T_n_1.x.array[:]
    T_n_1.x.array[:] = T_h.x.array[:]

try: 
    u_file.close() 
    A_file.close() 
    T_file.close() 
    P_file.close() 
except NameError: 
    pass 
