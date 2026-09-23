# Add an insulating layer
# Include only the full solver
# Add topography to full_code.py

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
                                 create_vector_block, set_bc,create_matrix_block,create_matrix,interpolation_matrix,discrete_gradient)
from dolfinx.fem.petsc import LinearProblem
import gc
import basix
import adios4dolfinx as adx 
from adios4dolfinx.adios2_helpers import (
    ADIOSFile,
    adios_to_numpy_dtype,
    resolve_adios_scope,
)
import adios2

if MPI.COMM_WORLD.Get_rank() ==0:   
    print(f'喊话：记录这个算例是测试 高斯地形的收敛性结果测试',flush=True) 

#####Function definitions
    

def apply_gaussian_cmb_topography(msh):
    """变形流体球壳及外部绝缘层，保持 ICB 和最外边界不动。"""

    if msh.geometry.dim != 3:
        raise RuntimeError(
            "Gaussian CMB topography requires a 3-D mesh."
        )

    r_outer = 5.0 * ro

    if not (0.0 < ri < ro < r_outer):
        raise ValueError("Require 0 < ri < ro < r_outer.")

    if beta_inverse < 0.0:
        raise ValueError("beta_inverse must be nonnegative.")

    # Allow both epsilon > 0 (inward depression) and epsilon < 0 (outward bulge)
    if not (ri < ro - epsilon < r_outer):
        raise RuntimeError(
            "Deformed CMB intersects the ICB or outer boundary: "
            "reduce the magnitude of epsilon."
        )

    coordinates = msh.geometry.x[:, :3]
    radius = np.linalg.norm(coordinates, axis=1)

    nonzero = radius > 1.0e-14

    radial_direction = np.zeros_like(coordinates)
    radial_direction[nonzero] = (
        coordinates[nonzero] / radius[nonzero, np.newaxis]
    )

    # Direction of the Gaussian-topography center: theta0 is colatitude and phi0 is longitude
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
        (radial_direction - topography_direction[None, :]) ** 2,
        axis=1,
    )

    h_topography = np.exp(-beta_inverse * distance_squared)

    # Radial-displacement weights: 0 at the ICB, 1 at the CMB, and 0 at the outermost boundary
    weight = np.zeros_like(radius)

    fluid = (radius >= ri) & (radius <= ro)
    insulator = (radius > ro) & (radius <= r_outer)

    weight[fluid] = (
        (radius[fluid] - ri) / (ro - ri)
    )

    weight[insulator] = (
        (r_outer - radius[insulator]) / (r_outer - ro)
    )

    new_radius = radius - epsilon * h_topography * weight

    coordinates[nonzero] = (
        radial_direction[nonzero]
        * new_radius[nonzero, np.newaxis]
    )

def norm_L2(comm, v ,measure):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(v, v) * measure)), op=MPI.SUM))

def norm_L2_B(comm, v):
    # """Compute the L2(Ω)-norm of B""" input is A not B 
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(curl(v), curl(v)) * dx)), op=MPI.SUM))

def norm_ds(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(v, v) * ds)), op=MPI.SUM))

def monitor_residual(ksp, its, rnorm):
    if  MPI.COMM_WORLD.Get_rank() ==0:
        print(f'Iteration {its}, Residual norm {rnorm}', flush=True)

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

    R = x[0]**2 + x[1]**2 + x[2]**2
    active = (R <= ro**2) & (R >= ri**2)

    # Compute only inside the fluid shell to avoid producing NaN first at r = 0 in the inner-core center
    xx = x[0, active]
    yy = x[1, active]
    zz = x[2, active]

    r2 = xx**2 + yy**2 + zz**2
    r = np.sqrt(r2)
    rho2 = xx**2 + yy**2

    poly = (
        -48 * ri * ro
        + 6 * (4 * ro + ri * (4 + 3 * ro)) * r
        - 4 * (4 + 3 * (ri + ro)) * r2
        + 9 * r**3
    )

    Br = zz * 5.0 / 8.0 * poly / r2

    # Bt = ctheta * rho / r^2
    ctheta = -15.0 / 4.0 * (r - ri) * (r - ro) * (3 * r - 4)

    # Bphi = cphi * z * rho / r^2
    cphi = 15.0 / 4.0 * np.sin(np.pi * (r - ri))

    values[0, active] = (
        Br * xx / r
        + ctheta * zz * xx / r**3
        - cphi * zz * yy / r2
    )

    values[1, active] = (
        Br * yy / r
        + ctheta * zz * yy / r**3
        + cphi * zz * xx / r2
    )

    values[2, active] = (
        Br * zz / r
        - ctheta * rho2 / r**3
    )

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
###########

def creat_measure(integral_type,msh,dim,entitiesmarkers):
    entities_indices, entities_markers = [], []
    for (marker, locator) in entitiesmarkers:
        entities = mesh.locate_entities(msh, dim, locator)
        entities_indices.append(entities)
        entities_markers.append(np.full_like(entities, marker))
    entities_indices = np.hstack(entities_indices).astype(np.int32)
    entities_markers = np.hstack(entities_markers).astype(np.int32)
    sorted_entities = np.argsort(entities_indices)
    entities_tag = mesh.meshtags(msh, dim, entities_indices[sorted_entities], entities_markers[sorted_entities])
    if dim==msh.topology.dim-1:
        msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    return ufl.Measure(integral_type, domain=msh, subdomain_data=entities_tag),entities_tag

###Simulation parameters      
num_time_steps =1000        
t_end = 0.1       

have_innercore=True      
maxit=50 
#Control parameters     
Ekman=0.001    
Pm=5    
Ra=100    
Pr=1    
app_para=-100    
#Mesh parameters    
ro=20/13      
ri=7/13        
outside_radius_ratio=5    

# Oliver et al. (2025) Gaussian CMB topography parameters
epsilon = 0.2 
theta0 = 2.40 
phi0 = 0.0 
beta_inverse = 40.0 

tag="pcd"     

Vol=4/3*np.pi*(ro**3-ri**3)    

name="cp6n_0.41w_full.xdmf"  

read_mesh = io.XDMFFile(MPI.COMM_WORLD, name ,"r")   
msh=read_mesh.read_mesh(name="Grid")   

gdim = msh.geometry.dim   
fdim = msh.topology.dim - 1   

cells = mesh.locate_entities(msh, gdim, lambda x: (x[0]**2+x[1]**2+x[2]**2<=ro**2+0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ri**2-0.0001)) 
smsh, smsh_to_msh = mesh.create_submesh(msh, gdim, cells)[:2]    

cell_imap = msh.topology.index_map(msh.topology.dim)   
num_cells = cell_imap.size_local + cell_imap.num_ghosts   
msh_to_smsh = np.full(num_cells, -1)   
msh_to_smsh[smsh_to_msh] = np.arange(len(smsh_to_msh))   
entity_maps = {smsh: msh_to_smsh}   

if have_innercore==False: 
    ## 11 insulating layer; 12 fluid shell 
    domains=[(11,lambda x: x[0]**2+x[1]**2+x[2]**2>=ro**2-0.0001), 
             (12,lambda x: (x[0]**2+x[1]**2+x[2]**2<=ro**2+0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ri**2-0.0001))] 

    ## 1 CMB  2 ICB
    boundaries = [(1, lambda x: np.isclose(x[0]**2+x[1]**2+x[2]**2, (outside_radius_ratio*ro)**2)), 
                  (2, lambda x: np.isclose(x[0]**2+x[1]**2+x[2]**2, ro**2)), 
                  (3, lambda x: np.isclose(x[0]**2+x[1]**2+x[2]**2, ri**2))] 

    ## 21 insulating layer; 22 fluid shell
    interiors = [(21,lambda x: (x[0]**2+x[1]**2+x[2]**2<=(outside_radius_ratio*ro)**2-0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ro**2+0.0001) ),
                 (22,lambda x: (x[0]**2+x[1]**2+x[2]**2<=ro**2-0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ri**2+0.0001))]
else:
    ## 11 insulating layer; 12 fluid shell; 13 inner core 
    domains=[(11,lambda x: x[0]**2+x[1]**2+x[2]**2>=ro**2-0.0001),
             (12,lambda x: (x[0]**2+x[1]**2+x[2]**2<=ro**2+0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ri**2-0.0001)),
             (13,lambda x: x[0]**2+x[1]**2+x[2]**2<=ri**2+0.0001)]

    ## 1 CMB  2 ICB 
    boundaries = [(1, lambda x: np.isclose(x[0]**2+x[1]**2+x[2]**2, (outside_radius_ratio*ro)**2)),
                  (2, lambda x: np.isclose(x[0]**2+x[1]**2+x[2]**2, ro**2)),
                  (3, lambda x: np.isclose(x[0]**2+x[1]**2+x[2]**2, ri**2))]

    ## 21 insulating layer; 22 fluid shell; 23 inner core  
    interiors = [(21,lambda x: (x[0]**2+x[1]**2+x[2]**2<=(outside_radius_ratio*ro)**2-0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ro**2+0.0001) ),
                 (22,lambda x: (x[0]**2+x[1]**2+x[2]**2<=ro**2-0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ri**2+0.0001)),
                 (23,lambda x: x[0]**2+x[1]**2+x[2]**2<=ri**2-0.0001)]

#dx_oc=ufl.Measure("dx",domain=smsh)  
dx_oc,cell_tag_oc=creat_measure("dx",smsh,gdim,domains[1:3]) 
ds_oc,facet_tag_oc=creat_measure("ds",smsh,fdim,boundaries[1:3])  
dS_oc,interiors_tag_oc=creat_measure("dS",smsh,fdim,interiors[1:2])  

dx,cell_tag_oc=creat_measure("dx",msh,gdim,domains) 
ds,facet_tag=creat_measure("ds",msh,fdim,boundaries[0:1]) 

###Function spaces 
PP2 = element("Lagrange", smsh.basix_cell(), 2, shape=(msh.geometry.dim,))   
U_space=fem.functionspace(smsh,("CG",2,(gdim,)))  
P_space=fem.functionspace(smsh,("CG",1))  
B_space=fem.functionspace(msh,("N1curl",2))   
T_space=fem.functionspace(smsh,("CG",1))  

WW=fem.functionspace(msh, ("CG", 2, (gdim,)))
TT = fem.functionspace(msh, ("Lagrange", 1))
WW_oc=fem.functionspace(smsh, ("CG", 2, (gdim,)))
TT_oc= fem.functionspace(smsh, ("Lagrange", 1))

u,v=ufl.TrialFunction(U_space), ufl.TestFunction(U_space)
p,q=ufl.TrialFunction(P_space), ufl.TestFunction(P_space)
A,Av=ufl.TrialFunction(B_space), ufl.TestFunction(B_space)
T,Tv=ufl.TrialFunction(T_space), ufl.TestFunction(T_space)

mm_space=fem.functionspace(msh,("CG",2))   # Used to construct the auxiliary-space projection operator; scalar element
ww_space=fem.functionspace(msh,("CG",2,(gdim,)))   # Construct the ww auxiliary space; H1 vector element

m,mv=ufl.TrialFunction(mm_space), ufl.TestFunction(mm_space)  
w,wv=ufl.TrialFunction(ww_space), ufl.TestFunction(ww_space)  

# Used for AMS preconditioning
G = discrete_gradient(mm_space,B_space) 
G.assemble()  
#G.view()

Pi = interpolation_matrix(ww_space,B_space) 
Pi.assemble()   
#Pi.view()

u_offset = U_space.dofmap.index_map.size_local * U_space.dofmap.index_map_bs
p_offset = u_offset + P_space.dofmap.index_map.size_local * P_space.dofmap.index_map_bs
B_offset = p_offset + B_space.dofmap.index_map.size_local * B_space.dofmap.index_map_bs
T_offset = B_offset + T_space.dofmap.index_map.size_local * T_space.dofmap.index_map_bs


#region ##Boundary definitions
class BoundaryCondition_u():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = fem.Function(U_space)
            u_D.interpolate(values)
            facets = facet_tag_oc.find(marker)
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

class BoundaryCondition_T(): ##Boundary
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = fem.Function(T_space)
            u_D.interpolate(values)
            #facets = facet_tag.find(marker)
            dofs = fem.locate_dofs_geometrical(T_space, boundaries[marker-1][1])
#            fem.locate_dofs_topological((W.sub(2),W2), fdim, facets)
            
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
            u_D = fem.Function(B_space)
            u_D.interpolate(values)
            facets = facet_tag.find(marker)
            dofs = fem.locate_dofs_topological(B_space, fdim, facets)
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

boundary_conditions_u = [BoundaryCondition_u("Dirichlet", 2, noslip),
                         BoundaryCondition_u("Dirichlet", 3, noslip)]

boundary_conditions_T = [BoundaryCondition_T("Dirichlet", 2, boundT_0),
                         BoundaryCondition_T("Dirichlet", 3, boundT_1)]

def boundary_out(x):
    return np.isclose(
        x[0]**2 + x[1]**2 + x[2]**2,
        (outside_radius_ratio * ro)**2,
    )

B_D = fem.Function(B_space)  
B_D.interpolate(boundA)  

# Reuse outer-boundary tags directly to avoid roundoff errors from coordinate-based relocation
facets = facet_tag.find(1)

dofs = fem.locate_dofs_topological(B_space, 2, facets)
B_d= fem.Function(B_space)
B_d.interpolate(boundA)
bcs3 = fem.dirichletbc(B_d, dofs)
bcs = []

for condition in boundary_conditions_u:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)

for condition in boundary_conditions_T:
    if condition.type == "Dirichlet":
        bcs.append(condition.bc)

bcs.append(bcs3)

# -------------------------------------Subcommunicator construction--------------------------------------
msh_n=smsh  # Pass the submesh
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

comm_n = MPI.COMM_WORLD
rank_n = comm_n.rank
size_n = comm_n.size

#solver_comm 
m=2  #Number of subproblem processes; update when tuning parameters
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
    msh_mm = read_mesh0.read_mesh(name="Grid") 
    read_mesh0.close() 

    gdim = msh_mm.geometry.dim   
    fdim = msh_mm.topology.dim - 1  

    cellsm = mesh.locate_entities(msh_mm, gdim, lambda x: (x[0]**2+x[1]**2+x[2]**2<=ro**2+0.0001)&(x[0]**2+x[1]**2+x[2]**2>=ri**2-0.0001)) 
    msh_m ,s_to_s= mesh.create_submesh(msh_mm, gdim, cellsm)[:2]    

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
    dx_default = ufl.Measure("dx", domain=msh_m) 

    hm = CellDiameter(msh_m) 
    pm,qm=ufl.TrialFunction(Vm), ufl.TestFunction(Vm)  
    apm=inner(grad(pm), grad(qm)) * dx_default
    apm+=0.1/hm*inner(pm,qm)*ds_default  
    Mpm=fem.form(apm)  
    # Matrix and vector
    PPm = assemble_matrix(Mpm, bcs=[])   
    PPm.assemble()     
    Lm = fem.form(inner(fem.Constant(msh_m, 0.0), qm) * dx_default)
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

if rank_n==0:
    print('daozhele')

def create_compatible_initial_B(
    msh,
    B_space,
    scalar_space,
    domain_measure,
    outer_facets,
    initializer,
):
    """构造与显式绝缘层兼容的离散无散度初始磁场。

    首先把 benchmark 磁场写成 B_raw（流体壳外为零），然后求解

        (grad(chi), grad(q)) = (B_raw, grad(q)).

    最终取 B0 = B_raw - grad(chi)。因此在绝缘层中
    B0 = -grad(chi)，是一个无旋势场；同时 B0 在离散弱意义下
    满足无散度条件。chi 在最外边界取零，与当前最外层的
    n x B = 0 边界条件相容。
    """
    B_raw = fem.Function(B_space)
    B_raw.name = "B_initial_raw"
    B_raw.interpolate(initializer)
    B_raw.x.scatter_forward()

    chi = ufl.TrialFunction(scalar_space)
    q_chi = ufl.TestFunction(scalar_space)

    a_chi = inner(grad(chi), grad(q_chi)) * domain_measure
    L_chi = inner(B_raw, grad(q_chi)) * domain_measure

    chi_zero = fem.Function(scalar_space)
    chi_zero.x.array[:] = 0.0
    chi_dofs = fem.locate_dofs_topological(
        scalar_space, msh.topology.dim - 1, outer_facets
    )
    bc_chi = fem.dirichletbc(chi_zero, chi_dofs)

    projection_problem = LinearProblem(
        a_chi,
        L_chi,
        bcs=[bc_chi],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "gamg",
            "ksp_rtol": 1.0e-10,
            "ksp_atol": 1.0e-12,
            "ksp_error_if_not_converged": True,
        },
    )
    chi_h = projection_problem.solve()
    chi_h.name = "initial_magnetic_potential_correction"
    chi_h.x.scatter_forward()

    # CG2 -> N1curl2 belongs to the same discrete de Rham sequence, so grad(chi_h)
    # can be interpolated directly into the magnetic-field space.
    grad_chi = fem.Function(B_space)
    grad_chi_expr = fem.Expression(
        grad(chi_h), B_space.element.interpolation_points()
    )
    grad_chi.interpolate(grad_chi_expr)
    grad_chi.x.scatter_forward()

    B0 = fem.Function(B_space)
    B0.name = "B_initial_compatible"
    B0.x.array[:] = B_raw.x.array - grad_chi.x.array
    B0.x.scatter_forward()

    if msh.comm.rank == 0:
        print(
            "Initial magnetic Helmholtz projection: "
            f"{projection_problem.solver.getIterationNumber()} CG iterations",
            flush=True,
        )

    return B0

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

###Define initial values
u_n_1=fem.Function(U_space)
u_n_2=fem.Function(U_space)
u_D = fem.Function(U_space)

A_n_1=fem.Function(B_space)
A_n_2=fem.Function(B_space)

B_initial = create_compatible_initial_B(
    msh=msh,
    B_space=B_space,
    scalar_space=mm_space,
    domain_measure=dx,
    outer_facets=facets,
    initializer=initial_B,
)

# Use the same insulating-layer-compatible initial value for both history levels of the second-order scheme
A_n_1.x.array[:] = B_initial.x.array
A_n_2.x.array[:] = B_initial.x.array
A_n_1.x.scatter_forward()
A_n_2.x.scatter_forward()


T_n_1=fem.Function(T_space)
T_n_2=fem.Function(T_space)
T_n_1.interpolate(initial_T)
T_n_2.interpolate(initial_T)
T_D=fem.Function(T_space)
T_D.interpolate(boundT_1)   #cell_tag_oc.find(12)
T_D0=fem.Function(T_space)

######       
###Weak forms        
delta_t = fem.Constant(msh, default_real_type(t_end / num_time_steps))
alpha = fem.Constant(smsh, default_real_type(6.0))

h = CellDiameter(smsh) 
n = FacetNormal(smsh) 
x = ufl.SpatialCoordinate(smsh) 

def mean(vn,vn_1): 
    return 1/2*(vn+vn_1) 

def star(v_n_1,v_n_2): 
    return 1/2*(3*v_n_1-v_n_2) 

def jump(phi, n): 
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-")) 

def jump2(phi):
    return phi("+") - phi("-")

def conv_u(w, z, v):
    return 0.5 * (
          inner(dot(grad(z), w), v)
        - inner(dot(grad(v), w), z)
    ) * dx_oc 

def conv_T(w, theta, Tv):
    return 0.5 * (
          inner(dot(w, grad(theta)), Tv)
        - inner(dot(w, grad(Tv)), theta)
    ) * dx_oc 


NT_n   = conv_T(u_n_1, T_n_1, Tv)
NT_nm1 = conv_T(u_n_2, T_n_2, Tv)

conv_TT = 1.5 * NT_n - 0.5 * NT_nm1

Nu_n   = conv_u(u_n_1, u_n_1, v)
Nu_nm1 = conv_u(u_n_2, u_n_2, v)

conv_uu = 1.5 * Nu_n - 0.5 * Nu_nm1


###  Navier-Stokes equations 
## Time-derivative term 
a_00=Ekman*(inner(u / delta_t, v) * dx_oc - inner(u_n_1 / delta_t, v) * dx_oc)     
##Weak boundary term for convection   
a_00+=Ekman*conv_uu
##Coriolis force   
a_00+=2*inner(cross(ufl.as_vector([0,0,1]),mean(u,u_n_1)),v)*dx_oc   

a_00+=Ekman * inner(div(u),div(v))*dx_oc
a_00+=Ekman * (inner(grad(mean(u,u_n_1)), grad(v)) * dx_oc)   
## Numerical stabilization term   
#F += 2*Ekman/delta_t *inner(div(u),div(v)) *dx   
zeros=fem.Constant(msh, default_real_type(0))   
# ## Pressure term  
a_01= -inner(p, div(v)) * dx_oc        
a_10= -inner(div(u), q) * dx_oc        
a_11= inner(zeros*p,q)*dx_oc          
## Lorentz force   
a_02  = -1.0/Pm * inner(cross(curl(mean(A,A_n_1)),star(A_n_1,A_n_2)),v)*dx_oc   
## Buoyancy term  
a_03 = -Ra/ro * inner(ufl.as_vector([x[0],x[1],x[2]])*star(T_n_1,T_n_2),v) *dx_oc     
a_03 +=Ra/ro * inner(ufl.as_vector([x[0],x[1],x[2]])*(ri*ro/norm_r(x)-ri),v) *dx_oc    

a_22 = inner(A/ delta_t,Av) * dx((11,12)) -inner(A_n_1/ delta_t,Av) * dx((11,12)) 
a_20 = inner(cross(curl(Av),star(A_n_1,A_n_2)),mean(u,u_n_1))*dx(12)     
a_22 += 1/Pm *inner(curl(mean(A,A_n_1)),curl(Av)) *dx((11,12))      

a_22 += inner(curl(A),curl(Av))*dx(13) 
a_22 += inner(div(A),div(Av))*dx((11,12)) 
##Time-derivative term     

a_33 =inner(T / delta_t, Tv) * dx_oc -inner(T_n_1 / delta_t, Tv) * dx_oc    
a_33 +=1/Pr*inner(grad(mean(T,T_n_1)),grad(Tv))*dx_oc   
a_33+=conv_TT

a_00_final=fem.form(ufl.lhs(a_00),entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)}) 
a_01_final=fem.form(ufl.lhs(a_01),entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)}) 
a_02_final=fem.form(ufl.lhs(a_02),entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)}) 
a_03_final=fem.form(ufl.lhs(a_03),entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)}) 
a_11_final=fem.form(a_11)  
a_10_final=fem.form(ufl.lhs(a_10))  

a_20_final=fem.form(ufl.lhs(a_20),entity_maps=entity_maps) 
a_22_final=fem.form(ufl.lhs(a_22),entity_maps=entity_maps) 

a_33_final=fem.form(ufl.lhs(a_33)) 

L0=fem.form(ufl.rhs(a_00)+ufl.rhs(a_01)+ufl.rhs(a_02)+ufl.rhs(a_03),entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)}) 

L1 = fem.form(inner(fem.Constant(smsh, 0.0), q) * dx_oc) 
L2=fem.form(ufl.rhs(a_20)+ufl.rhs(a_22),entity_maps=entity_maps) 
L3=fem.form(ufl.rhs(a_33)) 

a = [   
        [a_00_final, a_01_final, a_02_final,      None,    ], 
        [a_10_final, a_11_final,     None,        None,    ], 
        [a_20_final,     None,   a_22_final,      None,    ], 
        [   None,        None,       None,      a_33_final,], 
    ] 

L = [L0, L1, L2, L3]

ap=inner(grad(p), grad(q)) * dx_oc 
ap+=0.1/h*inner(p,q)*ds_oc 

Mp=fem.form(inner(p,q)*dx_oc,entity_maps=entity_maps)    
Ap=fem.form(ap,entity_maps=entity_maps)   

fp=2*Ekman/delta_t*inner(p,q)*dx_oc   
fp+=Ekman*inner(grad(p),grad(q))*dx_oc   

Fp=fem.form(fp,entity_maps=entity_maps) 


zero_dynamic = fem.Constant(smsh, default_scalar_type(0.0))
a_02_zero_final = fem.form(zero_dynamic * inner(A, v) * dx_oc,entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)})
a_20_zero_final = fem.form(zero_dynamic * inner(u, Av) * dx_oc,entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)})
a_33_zero_final = fem.form(zero_dynamic * inner(T, Tv) * dx_oc,entity_maps={msh: np.array(smsh_to_msh, dtype=np.int32)})


# Static matrix: zero-valued placeholders for dynamic coupling blocks preserve the sparsity pattern.
a_static = [
        [a_00_final, a_01_final,   a_02_zero_final, None,    ],
        [a_10_final, a_11_final,        None,      None,    ],
        [a_20_zero_final, None,       a_22_final,  None,    ],
        [   None,        None,            None,  a_33_final,],
    ]

a_dynamic_02_pattern = [
        [   None,          None,    a_02_zero_final,       None,],
        [   None,    a_11_final,               None,       None,],
        [a_20_zero_final,  None,               None,       None,],
        [   None,          None,               None, a_33_zero_final,],
    ]


a_p = ([   
        [a_00_final, a_01_final, a_02_final,     None,   ], 
        [   None,     a_11_final,       None,          None,   ], 
        [   None,     None,      a_22_final,      None,  ], 
        [   None,     None,       None,      a_33_final, ],  
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

_block_spaces = [U_space, P_space, B_space, T_space]
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
opts["ksp_atol"] = 1.0e-6      # Absolute residual tolerance  
opts["ksp_max_it"] =2000       # Set the maximum number of iterations to 800  
#opts["-pc_gasm_total_subdomains"] = 16 
ksp_g0.setFromOptions()   
#ksp_g0.setMonitor(monitor_residual)    

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
ksp_g2 = PETSc.KSP().create(msh.comm)  # type: ignore  
ksp_g2.setOperators(P2)    
ksp_g2.setType('gmres')    
ksp_g2.getPC().setType(PETSc.PC.Type.HYPRE)     #PETSc.PC.Type.JACOBI 
ksp_g2.getPC().setHYPREType('ams')              #boomeramg  parasails
ksp_g2.getPC().setHYPREDiscreteGradient(G)
ksp_g2.getPC().setHYPRESetInterpolations(3, None, None, Pi, None)
ksp_g2.getPC().setHYPRESetBetaPoissonMatrix(None)
opts = PETSc.Options()        # type: ignore 
opts["ksp_rtol"] = 1.0e-3     ##Residual tolerance  
opts["ksp_atol"] = 1.0e-6     # Absolute residual tolerance  
opts["ksp_max_it"] =2000       # Set the maximum number of iterations to 800  
opts["pc_hypre_ams_cycle_type"] =7 
ksp_g2.setFromOptions()   
#ksp_g2.setMonitor(monitor_residual)    

P3 = assemble_matrix(a_33_final, bcs=bcs)
P3.assemble()
b3 = create_vector(L3)
x3 = b3.duplicate() 
ksp_g3 = PETSc.KSP().create(msh.comm)  # type: ignore
ksp_g3.setOperators(P3) 
ksp_g3.setType('cg')  
ksp_g3.getPC().setType(PETSc.PC.Type.ASM) 
opts = PETSc.Options()  # type: ignore 
opts["ksp_rtol"] = 1.0e-3     # Residual tolerance 
opts["ksp_atol"] = 1.0e-6      # Absolute residual tolerance  
opts["ksp_max_it"] =2000       # Set the maximum number of iterations to 800  
ksp_g3.setFromOptions()    
#ksp_g3.setMonitor(monitor_residual)     

P01 = assemble_matrix(a_01_final, bcs=bcs)    
P01.assemble()    

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

b_tmp0 = b0.duplicate()
b_tmp1 = b1.duplicate()

comm = MPI.COMM_WORLD    
rank=MPI.COMM_WORLD.Get_rank()    
size = comm.Get_size()    

comm = MPI.COMM_WORLD    
rank=MPI.COMM_WORLD.Get_rank()    
size = comm.Get_size()    

t=0   
u_vis = fem.Function(WW_oc)   
P_vis = fem.Function(TT_oc)   
A_vis = fem.Function(WW)      
T_vis = fem.Function(TT_oc)    

tag='dd'
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
A_h=fem.Function(B_space) 
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
opts["ksp_atol"] = 1.0e-8     # Absolute residual tolerance      
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



def checkpoint_output_B(t=0.0):

    checkpoint_file = f"checkpoint_41w_B.bp4" 
    
    checkpoint_time = float(t) 


    A_n_1.name = "B_n_1"
    A_n_2.name = "B_n_2"

    
   
    A_n_1.x.scatter_forward()
    A_n_2.x.scatter_forward()


    # Step 1: write the mesh 
    # write_mesh uses Write mode by default and creates or overwrites the file 
    adx.write_mesh(
        checkpoint_file,
        msh,
        engine="BP4",
        time=checkpoint_time
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

    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank == 0: 
        print("Checkpoint B 输出完成") 
        print(f"文件名：{checkpoint_file}") 
        print(f"保存时间：{checkpoint_time}")  
        print("包含变量:u, B, T") 


def checkpoint_output(t=0.0):
    
    checkpoint_file = f"checkpoint_41w_uT.bp4" 
    
    checkpoint_time = float(t) 

    # Set the function names stored in the checkpoint
    u_n_1.name = "u_n_1"
    u_n_2.name = "u_n_2"

    T_n_1.name = "T_n_1"
    T_n_2.name = "T_n_2"
    
    # Update parallel ghost degrees of freedom
    u_n_1.x.scatter_forward()
    u_n_2.x.scatter_forward()

    T_n_1.x.scatter_forward()
    T_n_2.x.scatter_forward()

    # Step 1: write the mesh 
    # write_mesh uses Write mode by default and creates or overwrites the file 
    adx.write_mesh(
        checkpoint_file,
        smsh,
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
        print("Checkpoint_U+T 输出完成") 
        print(f"文件名：{checkpoint_file}") 
        print(f"保存时间：{checkpoint_time}")  
        print("包含变量:u, B, T") 

for n in range(1):

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
    A_h.x.array[:B_offset-p_offset]=x.array_r[p_offset:B_offset] 
    T_h.x.array[:T_offset-B_offset]=x.array_r[B_offset:T_offset] 
    
    u_h.x.scatter_forward()   
    p_h.x.scatter_forward()   
    A_h.x.scatter_forward()   
    T_h.x.scatter_forward()   
    
    #u_div=norm_L2(smsh.comm,div(u_h))
    #B_div=norm_L2(msh.comm,div(A_n_1))
    
    ekin=norm_L2(smsh.comm,u_h,dx_oc)
    div_B=norm_L2(msh.comm,div(A_h),dx((11,12)))
    print('磁场散度',div_B,flush=True)

    if MPI.COMM_WORLD.Get_rank() ==0: 
        with open("Ekin_"+tag+".txt","a") as ekinfile:     
            ekinfile.write(str(t)+'\t'+str(ekin)+'\n') 
        ekinfile.close() 
    
    if n % 1==0: 
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
        checkpoint_output_B(t) 
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

