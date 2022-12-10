import jax
import jax.numpy as jnp
import numpy as np

# time steps
ntime_steps = 10000
# target porosity
target_porosity = 0.4

@jax.jit
def soil_props(porosity):
    n = porosity
    # Thermal Conductivity, W/m-K
    lambda_soil = 1.0 
    lambda_water = 0.6 

    # Specific heat, J/kg-K
    cp_soil = 8000 
    cp_water = 4290 

    # Thermal properties of the soil-water medium
    lambda_medium = lambda_soil * (1 - n) + lambda_water * n #W/m-K
    cp = cp_soil * (1 - n) + cp_water * n #J/kg-K

    # Densities kg/m3
    rhow = 980 # water
    rhoS = 1850 # soil

    #Thermal diffusivity
    # alpha_water = lambda_water/(rhow * cp_water)
    # alpha_soil = lambda_soil/(rhoS * cp_soil)
    alpha = lambda_medium/(rhoS * cp) 

    # particle size m
    d50 = 0.025*0.001 
    # permeability m2
    permeability = (1/180)*((n**3)/((1-n)**2))*(d50**2)
    
    return alpha, permeability


@jax.jit
def conduction_convection(permeability, porosity, alpha):
    # box size, m
    w = h = 1
    # intervals in x-, y- directions, m
    dx = dy = 0.005
    # rho water kg/m3
    rhow = 980 
    # Viscosity kg/m-s
    mu = 1.00E-03 
    # gravity
    g = 9.81 #m/s2
    # Thermal expansion 
    beta = 8.80E-05

    # Set conduction to 0 to disable
    conduction = 1.
    convection = 1.

    # Temperature of the cable
    Tcool, Thot = 0, 30

    # pipe geometry
    pr, px, py = 0.0125, 0.5, 0.5
    pr2 = pr**2

    # Calculations
    nx, ny = int(w/dx),int(h/dy)
    dx2, dy2 = dx*dx, dy*dy

    # Time step
    dt = 0.5 

    # nsteps
    nsteps = ntime_steps

    # Compute heat flow based on permeability
    u0 = jnp.zeros((nx, ny))

    mask_cable = np.zeros((200, 200))
    # Initial conditions
    for i in range(97,103):
        for j in range(97,103):
            if ((i*dx-px)**2 + (j*dy-py)**2) <= pr2:
                mask_cable[i,j] = 1.0
    mask_cable = jnp.asarray(mask_cable)
    u0 = mask_cable * Thot

    # Apply zero temp at boundaries
    mask_boundaries = np.ones((200, 200))
    mask_boundaries[:,0] = 0.0
    mask_boundaries[:,199] = 0.0
    mask_boundaries[0,:] = 0.0
    mask_boundaries[199,:] = 0.0
    mask_boundaries = jnp.asarray(mask_boundaries)

    u0 = mask_boundaries * u0
    
    # Copy to u
    u = u0

    convection_factor = convection * dt * permeability * (1 / (porosity * mu) * g * rhow) / dy

    def step(i, carry):
        u0, u = carry
        uip = jnp.roll(u0, 1, axis=0)
        ujp = jnp.roll(u0, 1, axis=1)
        uin = jnp.roll(u0, -1, axis=0)
        ujn = jnp.roll(u0, -1, axis=1)
        u = u0 + conduction * dt * alpha * ((uin -2 * u0 + uip)/dy2 + (ujn - 2 * u0 + ujp)/dx2) + (uip - u0) * convection_factor * (1 - beta * u0)

        # Apply initial conditions    
        u0 = mask_cable * Thot
        u0 = mask_boundaries * u0

        # Set u0 as u    
        u0 = u
        return (u0, u)

    # Iterate
    u0, u = jax.lax.fori_loop(0, nsteps, step, (u0, u))

    return u

@jax.jit
def heat_transfer(permeability, porosity, alpha, target_u):
    u = conduction_convection(permeability, porosity, alpha)
    return jnp.linalg.norm(u - target_u)


# Compute targets
target_alpha, target_permeability = soil_props(target_porosity)
print("Target permeability: ", target_permeability, " porosity: ", target_porosity)
uzeros = jnp.zeros((200, 200))
target_u = conduction_convection(target_permeability, target_porosity, target_alpha)
print("Compute target completed, norm target temp: ", jnp.linalg.norm(target_u))

# Newton Raphson iteration for solving the inverse problem.
permeability_factor = 0.005
porosity = 0.45
permeability_tolerance = 1e-10
porosity_tolerance = 1e-15

alpha, perm = soil_props(porosity)
# particle size m
d50 = 0.025 * 0.001

permeability = permeability_factor * target_permeability
# permeability.requires_grad = True

for i in range(0, 50):
    # Function of heat loss
    @jax.jit
    def compute_loss(permeability):
        return heat_transfer(permeability, porosity, alpha, target_u)

    # Gradient of heat loss
    heat_grad = jax.grad(compute_loss)

    f = compute_loss(permeability)
    df = heat_grad(permeability)
    h = f/df
    print(i, " Permeability: ", permeability, " df: ", df, " f: ", f, " h: ", h)
    
    permeability = permeability - h
    k = permeability
    
    for j in range(1,10):
        fn = ((d50**2)/180) * ((porosity**3) / (1 - porosity)**2) - k
        dfn = ((d50**2)/180) * (porosity**4 - 4 * porosity**3 + 3 * porosity**2)/((1 - porosity)**4)
        print(i, " fn: ", fn, "\tdfn: ", dfn, " \tporosity:", porosity)
        if abs(fn) < porosity_tolerance:
            break
        porosity = porosity - fn/dfn
        alpha, perm = soil_props(porosity)
    
    # Check if permeability loss term is less than 1e-3
    if abs(f) < 1e-3:
        break