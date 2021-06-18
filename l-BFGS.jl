using Printf
@printf "Initializing...\n"

using LinearAlgebra, JLD2
@everywhere include("code/acoustic_solver.jl")
@everywhere include("code/acoustic_solver_parallel.jl")
@everywhere include("code/adjoint_method.jl")
@everywhere include("code/total_variation.jl")
@everywhere include("code/projection_fn.jl")
@everywhere include("code/optimization.jl")
@everywhere include("code/l-BFGS_code.jl")

@printf "Loading...\n"
@load "/Users/lida/Desktop/Proj_grad_method/temp_data/data.jld2"
@load "/Users/lida/Desktop/Proj_grad_method/model_data/overthrust_50_76_by_251.jld2"
# u0 = copy(c_gauss_5)
u0 = copy(c)

@printf "Preparing optimization function handle...\n"
# Initialization: function handle
eval_fn_handle(x) = eval_obj_fn(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
eval_grad_handle(x) = compute_gradient(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

# eval_fn_handle(x) = eval_obj_fn_mc(received_data, x, 2, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
# eval_grad_handle(x) = compute_gradient_mc(received_data, x, 2, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

# u0 = zeros(Nx, Ny, 2)
# u0[:,:,1] = c
# u0[:,:,2] = rho;
# @load "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_result.jld2"
# u0 = copy(u1)

# Initialization: optimization coef
m = 5
eta = 0.5
iterMax = 20
wwp_c1 = 1e-5
wwp_c2 = 0.95

# Optimization main function
u1, obj_fn = l_BFGS(u0, eval_fn_handle, eval_grad_handle; m=m, eta=eta, iterMax=iterMax, wwp_c1=wwp_c1, wwp_c2=wwp_c2, LinesearchMax=10)

# Saving
@printf "\nSaving...\n"
@save "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_result.jld2" u1 obj_fn
@printf "Saved.\n"