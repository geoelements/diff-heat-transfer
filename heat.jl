# using Plots
using LinearAlgebra
using ForwardDiff
using DelimitedFiles
using Statistics

# time steps
ntime_steps = 10000
# target porosity
target_porosity = 0.4
# Natural soil properties
const clay_alpha = 1.56868e-7
const clay_permeability = 1e-16
const clay_porosity = 0.6 

function soil_props(porosity)
    n = porosity
    # Thermal Conductivity, W/m-K
    lambda_soil = 1.0 
    lambda_water = 0.6 

    # Specific heat, J/kg-K
    cp_soil = 8000 
    cp_water = 4290 

    # Thermal properties of the soil-water medium
    lambda = lambda_soil * (1 - n) + lambda_water * n #W/m-K
    cp = cp_soil * (1 - n) + cp_water * n #J/kg-K

    # Densities kg/m3
    rhow = 980 # water
    rhoS = 1850 # soil

    #Thermal diffusivity
    alpha_water = lambda_water/(rhow * cp_water)
    alpha_soil = lambda_soil/(rhoS * cp_soil)
    alpha = lambda/(rhoS * cp) 

    # particle size m
    d50 = 0.025*0.001 
    # permeability m2
    permeability = (1/180)*((n^3)/((1-n)^2))*(d50^2)
    
    return alpha, permeability
end

function conduction_convection(permeability, porosity, alpha)
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
    pr2 = pr^2

    # Calculations
    nx, ny = convert(Int64, w/dx), convert(Int64, h/dy)
    dx2, dy2 = dx*dx, dy*dy

    # Time step
    dt = 0.5 

    # nsteps
    nsteps = ntime_steps

    # Compute heat flow based on permeability
    u0 = zeros(nx, ny) * permeability / permeability
    
    # Initial conditions
    alphas = ones(nx, ny) * clay_alpha * (alpha / alpha)
    perm = ones(nx, ny) * clay_permeability * (permeability / permeability)
    poros= ones(nx, ny) * clay_porosity * (porosity / porosity)
    # Initial conditions
    for i = 50:150
        for j = 50:150
            alphas[i,j] = alpha
            perm[i,j] = permeability
            poros[i,j] = porosity
            # Cable
            if ((i*dx-px)^2 + (j*dy-py)^2) <= pr2
                u0[i,j] = Thot
            end
        end
    end

    # Copy to u
    u = deepcopy(u0)

    convection_factor = convection * dt * ((1 / mu) * g * rhow) / dy

    # Iterate
    for k = 1:nsteps
        for i = 2:nx-1
            for j = 2:ny-1
                u[i, j] = u0[i, j] +
                    conduction * dt * alphas[i, j] * ((u0[i+1, j] - 2 * u0[i,j] + u0[i-1, j])/dy2 + 
                                               (u0[i, j+1] - 2 * u0[i,j] + u0[i, j-1])/dx2) + 
                    (u0[i-1,j] - u0[i,j]) * convection_factor * perm[i, j] * (1 - beta * u0[i,j]) / poros[i,j]
            end
        end
        # Initial conditions
        for i = 97:103
            for j = 97:103
                if ((i*dx-px)^2 + (j*dy-py)^2) <= pr2
                    u[i,j] = Thot
                end
            end
        end

        u0 = copy(u)
    end
    return u
end

function heat_transfer(permeability, porosity, alpha, target_u)
    u = conduction_convection(permeability, porosity, alpha)
    return norm(u - target_u)
end

# Compute targets
target_alpha, target_permeability = soil_props(target_porosity)
println("Target permeability: ", target_permeability, " porosity: ", target_porosity)
uzeros = zeros(200, 200)
target_u = conduction_convection(target_permeability, target_porosity, target_alpha)
println("Compute target completed, norm target temp: ", norm(target_u))

# Newton Raphson iteration for solving the inverse problem.
permeability_factor = 0.005
porosity = 0.45
permeability_tolerance = 1e-3
porosity_tolerance = 1e-15

alpha, perm = soil_props(porosity)
# particle size m
d50 = 0.025 * 0.001
println("Permeability: ", perm)

permeability = permeability_factor * target_permeability

# Partial derivative of heat wrt permeability factor
∂heat_∂permeability(permeability, porosity, alpha, target_u) = 
    ForwardDiff.derivative(permeability -> heat_transfer(permeability, porosity, alpha, target_u), permeability)

# Iterate to update permeability
for i = 1:50
    f = heat_transfer(permeability, porosity, alpha, target_u)
    df = ∂heat_∂permeability(permeability, porosity, alpha, target_u)
    h = f/df
    println(i, " Permeability: ", permeability, " df: ", df, " f: ", f, " h: ", h)
    if abs(f) < permeability_tolerance
        break
    end
    global permeability = permeability - h
    
    # Iterate to update porosity
    k = permeability
    for i = 1:10
        fn = ((d50^2)/180) * ((porosity^3) / (1 - porosity)^2) - k
        dfn = ((d50^2)/180) * (porosity^4 - 4 * porosity^3 + 3 * porosity^2)/((1 - porosity)^4)
        println(i, " fn: ", fn, "\tdfn: ", dfn, " \tporosity:", porosity)
        if abs(fn) < porosity_tolerance
           break
        end
        global porosity = porosity - fn/dfn
        global alpha, perm = soil_props(porosity)
    end
end

temp = conduction_convection(permeability, porosity, alpha)
writedlm("u0.csv",  temp, ',')
println("Norm of heat: ", norm(temp))