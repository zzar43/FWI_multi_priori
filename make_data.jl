using LinearAlgebra, ImageFiltering, JLD2, Printf
using Distributed, SharedArrays

@everywhere include("code/acoustic_solver.jl")
@everywhere include("code/acoustic_solver_parallel.jl")
@everywhere include("code/adjoint_method.jl")
@everywhere include("code/total_variation.jl")
@everywhere include("code/projection_fn.jl")
@everywhere include("code/optimization.jl")

# make data
@load "model_data/overthrust_50_76_by_251.jld2"
c_true = c_true
h = 50 / 1000
Nx, Ny = size(c_true)
rho_true = ones(Nx, Ny)
# c = copy(c_linear_30)
c = imfilter(c_true, Kernel.gaussian(5));
c = (c .- minimum(c)) ./ (maximum(c)-minimum(c)) * (maximum(c_true)-minimum(c_true)) .+ minimum(c_true);
c[1:60, 160:164] .= c_true[1:60, 160:164];
rho = copy(rho_true)

Fs = 250
dt = 1/Fs
Nt = 1000
t = range(0, step=dt, length=Nt)

# source
source_fre = 5
source = source_ricker(source_fre,0.25,t)
source_num = 10
source_position = zeros(Int,source_num,2)
for i = 1:source_num
    source_position[i,:] = [3, 12+25(i-1)]
end
source = repeat(source, 1, 1)

# receiver
receiver_num = 126
receiver_position = zeros(Int,receiver_num,2)
for i = 1:receiver_num
    receiver_position[i,:] = [1, (i-1)*2+1]
end

# PML
pml_len = 30
pml_coef = 50
@printf "Nx = %d, Ny = %d\n" Nx Ny
@printf "Fs = %d, Nt = %d, source frequency: %d Hz\n" Fs Nt source_fre
@printf "Source number: %d, receiver number: %d\n" source_num receiver_num
@printf "PML thick: %d, PML coef = %d\n" pml_len pml_coef

# make data
println("Computing received data.")
@time received_data = multi_solver_no_wavefield(c_true, rho_true, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
println("Received data done.")

println("Saving.")
@save "temp_data/data.jld2" Nx Ny h c c_true rho rho_true Fs dt Nt t source source_num source_position receiver_num receiver_position pml_len pml_coef received_data
println("Saved.")