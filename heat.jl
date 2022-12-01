# using Plots
using LinearAlgebra
using ForwardDiff
using DelimitedFiles
using Statistics

ntime_steps = 50000
# target porosity
n = 0.4

function soil_props(n)
    # Thermal Conductivity, W/m-K
    lambda_soil = 1.0 
    lambda_water = 0.6 

    # Specific heat, J/kg-K
    cp_soil = 8000 
    cp_water = 4290 

    # Thermal properties of the soil-water medium
    lambda = lambda_soil * (1-n) + lambda_water * n #W/m-K
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

function conduction_convection(permeability, alpha, nsteps)
    # box size, m
    w = h = 1
    # intervals in x-, y- directions, m
    dx = dy = 0.005
    # Porosity
    n = 0.45
    
    # Viscosity kg/m-s
    mu = 1.00E-03 
        
    # gravity
    g = 9.81 #m/s2
    
    # density of water
    rhow = 980 

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

    # ========================================#
    #              Compute Target             #
    # ========================================#
    t0 = Tcool * ones(nx, ny)
    # Initial conditions
    for i = 97:103
        for j = 97:103
            if ((i*dx-px)^2 + (j*dy-py)^2) <= pr2
                t0[i,j] = Thot
            end
        end
    end

    # Copy to u
    temp = deepcopy(t0)
    convection_factor = convection * dt * (1 / (n * mu) * permeability * g * rhow) / dy

    # Iterate
    for k = 1:nsteps
        for i = 2:nx-1
            for j = 2:ny-1
                temp[i, j] = t0[i, j] +
                    conduction * dt * alpha * ((t0[i+1, j] - 2 * t0[i,j] + t0[i-1, j])/dy2 + 
                                            (t0[i, j+1] - 2 * t0[i,j] + t0[i, j-1])/dx2) + 
                    (t0[i-1,j] - t0[i,j]) * convection_factor * (1 - beta * t0[i,j])
            end
        end
        # Initial conditions
        for i = 97:103
            for j = 97:103
                if ((i*dx-px)^2 + (j*dy-py)^2) <= pr2
                    temp[i,j] = Thot
                end
            end
        end
        t0 = copy(temp)
    end
    return temp
end

# Compute targets
alpha, target_permeability = soil_props(n)
println("Target permeability: ", target_permeability, " porosity: ", n)
target = conduction_convection(target_permeability, alpha, ntime_steps)
println("Compute target completed, norm target temp: ", norm(target))

function heat_transfer(permeability_factor, porosity, alpha, target_permeability)
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
    u0 = Tcool * ones(nx, ny) * permeability_factor
    
    # Initial conditions
    for i = 97:103
        for j = 97:103
            if ((i*dx-px)^2 + (j*dy-py)^2) <= pr2
                u0[i,j] = Thot
            end
        end
    end

    # Copy to u
    u = deepcopy(u0)

    convection_factor = convection * dt * 
            (1 / (porosity * mu) * target_permeability * permeability_factor * g * rhow) / dy

    # Iterate
    for k = 1:nsteps
        for i = 2:nx-1
            for j = 2:ny-1
                u[i, j] = u0[i, j] +
                    conduction * dt * alpha * ((u0[i+1, j] - 2 * u0[i,j] + u0[i-1, j])/dy2 + 
                                               (u0[i, j+1] - 2 * u0[i,j] + u0[i, j-1])/dx2) + 
                    (u0[i-1,j] - u0[i,j]) * convection_factor * (1 - beta * u0[i,j])
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
    return norm(u - target) / norm(target)
end

# Newton Raphson iteration for solving the inverse problem.
permeability_factor = 0.005
permeability_tolerance = 1e-10
porosity = 0.45
porosity_tolerance = 1e-15

alpha, permeability = soil_props(porosity)
# particle size m
d50 = 0.025 * 0.001

# Partial derivative of heat wrt permeability factor
∂heat_∂permeability(permeability_factor, porosity, alpha, target_permeability) = 
    ForwardDiff.derivative(permeability_factor -> heat_transfer(permeability_factor, porosity, alpha, target_permeability), permeability_factor)

# Iterate to update permeability
for i = 1:50
    df = ∂heat_∂permeability(permeability_factor, porosity, alpha, target_permeability)
    f = heat_transfer(permeability_factor, porosity, alpha, target_permeability)
    println(i, " Permeability: ", permeability_factor * target_permeability, " df: ", df, " f: ", f, " h: ", f/df)
    if abs(f) < permeability_tolerance
        break
    end
    global permeability_factor = permeability_factor - f/df
    
    # Iterate to update porosity
    k = permeability_factor * target_permeability
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

temp = conduction_convection(permeability_factor*target_permeability, alpha, ntime_steps)
writedlm("u0.csv",  temp, ',')
println("Norm of heat: ", norm(temp))
