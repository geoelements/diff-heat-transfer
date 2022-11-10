# using Plots
using LinearAlgebra
using ForwardDiff
using DelimitedFiles
using Statistics

target_permeability = 1e-12
ntime_steps = 100000

function conduction_convection(permeability = 1.0E-12, nsteps=10000)
    # box size, m
    w = h = 1
    # intervals in x-, y- directions, m
    dx = dy = 0.005
    # Thermal conductivity W/(m-K)
    thermal_conductivity = 1.602 
    # Thermal diffusivity, m2.s-1
    alphaSoil =  1.97e-7  # m^2/s
    alphaSteel = 2.3e-5   # m^2/s
    alphaWater = 1.39e-7  # m^2/s
    # Porosity
    n = 0.45

    # Viscosity kg/m
    mu = 1.00E-03 
    
    # Thermal expansion 
    beta = 8.80E-05
    # Cf
    cf = 4290
    # rhow
    rhow = 980
    # gravity
    g = 9.81 

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
    convection_factor = convection * dt * (1/(n*mu)*permeability*g*rhow) / dy
    println("convection factor ", convection_factor)
    # Iterate
    for k = 1:nsteps
        for i = 2:nx-1
            for j = 2:ny-1
                # The velocity corresponds to differential density, since we are measuring the differnetial temp,
                # the rho(1 - beta(T)) is written as rho*(beta*DeltaT) cable
                temp[i, j] = t0[i, j] +
                    conduction * dt * alphaSoil * ((t0[i+1, j] - 2 * t0[i,j] + t0[i-1, j])/dy2 + 
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

target = conduction_convection(target_permeability,100000)

function heat_transfer(soil_properties)
    permeability_factor = soil_properties[1]
    # box size, m
    w = h = 1
    # intervals in x-, y- directions, m
    dx = dy = 0.005
    # Thermal conductivity W/(m-K)
    thermal_conductivity = 1.602 
    # Thermal diffusivity, m2.s-1
    alphaSoil =  1.97e-7  # m^2/s
    alphaSteel = 2.3e-5   # m^2/s
    alphaWater = 1.39e-7  # m^2/s
    # Porosity
    n = soil_properties[2] # 0.45

    # Viscosity kg/m
    mu = 1.00E-03 
    # Permeability m2
    permeability_target = 1.0E-12
    # Thermal expansion 
    beta = 8.80E-05
    # Cf
    cf = 4290
    # rhow
    rhow = 980
    # gravity
    g = 9.81 

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

    # ========================================#
    #         Compute Heat Transfer           #
    # ========================================#
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
    
    convection_factor = convection * dt * (1/(n*mu) * permeability_target * permeability_factor * g * rhow) / dy
    # Iterate
    for k = 1:nsteps
        for i = 2:nx-1
            for j = 2:ny-1
                # The velocity corresponds to differential density, since we are measuring the differnetial temp,
                # the rho(1 - beta(T)) is written as rho*(beta*DeltaT) cable
                u[i, j] = u0[i, j] +
                    conduction * dt * alphaSoil * ((u0[i+1, j] - 2 * u0[i,j] + u0[i-1, j])/dy2 + 
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
tolerance = 1e-10

# Initialize soil properties
soil_properties = zeros(2)
soil_properties[1] = permeability_factor
soil_properties[2] = 0.45

# Iterate to update permeability
for i = 1:50
    df = ForwardDiff.gradient(heat_transfer, soil_properties)[1]
    f = heat_transfer(soil_properties)
    println(i, " Permeability: ", permeability_factor * 1e-12, " df: ", df, " f: ", f, " h: ", f/df)
    if abs(f) < tolerance
        break
    end
    global permeability_factor = permeability_factor - f/df
    global soil_properties[1] = permeability_factor
end

temp = conduction_convection(permeability_factor*target_permeability, ntime_steps)
writedlm("u0.csv",  temp, ',')
println("Norm of heat: ", norm(temp))