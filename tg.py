import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Box size
L0 = 2.0*np.pi
Nx, Ny = 4, 4

# Dimensionless quantities
Re = 1.0e3

# Time-integration controls
dealias = 3/2
stop_sim_time = 20
timestepper = d3.RK443
max_timestep = 1e-1
dtype = np.float64

# Substitutions for coefficients
coef_visc = 1.0/Re 

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-L0/2, L0/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-L0/2, L0/2), dealias=dealias)

# Fields for realisation 1
u = dist.Field(name='u', bases=(xbasis,ybasis))
v = dist.Field(name='v', bases=(xbasis,ybasis))
p = dist.Field(name='p', bases=(xbasis,ybasis))
vel = dist.VectorField(coords, name='vel', bases=(xbasis,ybasis))
tau_p = dist.Field(name='tau_p')  # Penalty term for incompressibility


# Define x and y gradients
dx = lambda A: d3.Differentiate(A, d3.Coordinate('x'))
dy = lambda A: d3.Differentiate(A, d3.Coordinate('y'))

# Define coordinates, vectors and body force
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

# Problem definition ==============================================================================
problem = d3.IVP([u,v,p,tau_p,vel], namespace=locals())

problem.add_equation("dt(u) + dx(p) - coef_visc*lap(u) = -u*dx(u) - v*dy(u)")
problem.add_equation("dt(v) + dy(p) - coef_visc*lap(v) = -u*dx(v) - v*dy(v)")                
        
problem.add_equation("dx(u) + dy(v) + tau_p = 0")  # Divergence free condition
problem.add_equation("integ(p) = 0") # Mean pressure =0
problem.add_equation("vel - u*ex - v*ey = 0") # Define velocity vector


# =================================================================================================

# Solver ==========================================================================================
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
# =================================================================================================

# Initial conditions ==============================================================================
u['g'] = -np.cos(2*np.pi*x/L0)*np.sin(2*np.pi*y/L0) # taylor-green initial conditions
v['g'] = np.sin(2*np.pi*x/L0)*np.cos(2*np.pi*y/L0)

# =================================================================================================

# CFL =============================================================================================
CFL = d3.CFL(solver, initial_dt=0.01*max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(vel)
# =================================================================================================


# Snapshots output ================================================================================
# Fields for visualisation
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1.0, max_writes=10)
snapshots.add_task(dx(v)-dy(u), name='vorticity')
snapshots.add_task(u, name='u')
snapshots.add_task(v, name='v')

# Statistical analysis ============================================================================
# Mostly volume averaged quantities to explore evolution of uncertainty
analysis = solver.evaluator.add_file_handler('analysis', sim_dt=0.01, max_writes=400000)

# Kinetic energy of each field
analysis.add_task(d3.Average((0.5*(u*u + v*v)), ('x', 'y')), layout='g',name='<E>')

ft_2 = np.exp(-4*0.001*solver.sim_time)
u_0 = -np.cos(2*np.pi*x/L0)*np.sin(2*np.pi*y/L0)*ft_2
v_0 = np.sin(2*np.pi*x/L0)*np.cos(2*np.pi*y/L0)*ft_2

analysis.add_task(d3.Average((np.sqrt(u_0*u_0 + v_0*v_0) - np.sqrt((u)*(u) + (v)*(v)))), layout='g',name='<error>')


# Flow properties - primarily for logger ==========================================================
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u)*(u)+(v)*(v),name='E')
# =================================================================================================

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_E = flow.max('E')

            """
            ft_2 = np.exp(-4*0.001*solver.sim_time)
            u_0 = -np.cos(2*np.pi*x/L0)*np.sin(2*np.pi*y/L0)*ft_2
            v_0 = np.sin(2*np.pi*x/L0)*np.cos(2*np.pi*y/L0)*ft_2

            print(u_0)
            error = (np.sqrt(u_0*u_0 + v_0*v_0) - np.sqrt((u)*(u) + (v)*(v)))
            l2_error = d3.Average(error, ('x', 'y'))

            """

            logger.info('Time=%e, dt=%e, max(E)=%e' %(solver.sim_time, timestep, max_E) )        
            
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
