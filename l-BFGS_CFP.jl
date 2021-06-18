using Printf
@printf "Initializing...\n"

using LinearAlgebra, JLD2
@everywhere include("code/acoustic_solver.jl")
@everywhere include("code/acoustic_solver_parallel.jl")
@everywhere include("code/adjoint_method.jl")
@everywhere include("code/total_variation.jl")
@everywhere include("code/projection_fn.jl")
@everywhere include("code/optimization.jl")
@everywhere include("code/l-BFGS_CFP_code.jl")

@printf "Loading...\n"
@load "/Users/lida/Desktop/Proj_grad_method/temp_data/data.jld2"
u_true = copy(c_true)
u0 = copy(c)
@load "/Users/lida/Desktop/Proj_grad_method/model_data/overthrust_50_76_by_251.jld2"
u0 = copy(c_gauss_5)
# @load "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_temp_box_tv_30.jld2"
# u0 = copy(u1)

# Initialization: function handle
@printf "Preparing optimization function handle...\n"
eval_fn_handle(x) = eval_obj_fn(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
eval_grad_handle(x) = compute_gradient(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);


# eval_fn_handle(x) = eval_obj_fn_mc(received_data, x, 2, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
# eval_grad_handle(x) = compute_gradient_mc(received_data, x, 2, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
# u0 = zeros(Nx, Ny, 2)
# u0[:,:,1] = c
# u0[:,:,2] = rho
# u_true = zeros(Nx, Ny, 2)
# u_true[:,:,1] = c_true
# u_true[:,:,2] = rho_true

# Initialization: quadratic CFP function handle
@printf "Preparing quadratic CFP function handle...\n"

# a = [minimum(c_true), minimum(rho_true)]
# b = [maximum(c_true), maximum(rho_true)]
# Nc = 2
# position_ind = ones(Nx, Ny, Nc)
# # position_ind[1:50,1:50,2] .= 1
# # position_ind[51:end,51:end,1] .= 1
# avg = [sum(u_true[:,:,1].*position_ind[:,:,1])./norm(position_ind[:,:,1])^2 sum(u_true[:,:,2].*position_ind[:,:,2])./norm(position_ind[:,:,2])^2]
# proj_box_handle(x) = proj_box_mc(x, Nx, Ny, Nc, a, b)
# proj_avg_handle(x) = proj_avg_mc(x, Nx, Ny, Nc, position_ind, avg)
# tv_tau = 1
# tv_tau = tv_tau * eval_vtv(u_true, Nx, Ny, Nc)
# eps_tv = 0.02 * eval_vtv(u_true, Nx, Ny, Nc)
# eps_box = 0.01
# eps_avg = 0.01
# iterMax_CFP = 500

box_min = minimum(c_true)
box_max = maximum(c_true)
# box_min = 0
# box_max = 10
box_threshold = 0.02

# tv_tau = 0.50 * eval_tv(c_true, Nx, Ny)
# tv_tau_threshold = 0.01 * eval_tv(c_true, Nx, Ny)

# tv_tau = 800
tv_tau = 1000
# tv_tau = 1200
tv_tau_threshold = 0.05 * tv_tau

# position_ind = zeros(Nx, Ny, 3)
# position_ind[:,:,1] = p1
# position_ind[:,:,2] = p2
# position_ind[:,:,3] = p3
# avg = zeros(3)
# avg[1] = sum(u_true.*position_ind[:,:,1])./norm(position_ind[:,:,1])^2
# avg[2] = sum(u_true.*position_ind[:,:,2])./norm(position_ind[:,:,2])^2
# avg[3] = sum(u_true.*position_ind[:,:,3])./norm(position_ind[:,:,3])^2

position_ind = copy(p3)
avg = sum(u_true.*position_ind)./norm(position_ind)^2

# position_ind = zeros(Nx, Ny, 2)
# position_ind[:,:,1] = p1
# position_ind[:,:,2] = p3
# avg = zeros(2)
# avg[1] = sum(u_true.*position_ind[:,:,1])./norm(position_ind[:,:,1])^2
# avg[2] = sum(u_true.*position_ind[:,:,2])./norm(position_ind[:,:,2])^2

proj_avg_handle(x) = proj_avg(x, Nx, Ny, position_ind, avg; N_avg=1)
avg_tau = 5
avg_tau_threshold = 0.5

# Phi = zeros(Nx*Ny, 25)
# for i = 1:5
#     for j = 1:5
#         phi = zeros(Nx, Ny)
#         phi[(i-1)*20+1:i*20, (j-1)*20+1:j*20] .= 1
#         phi = phi / norm(phi)
#         Phi[:,(i-1)*5+j] = reshape(phi, Nx*Ny, 1)
#     end
# end
# l1_tau = 0.8
# u_ref = copy(c)
# l1_tau = l1_tau * eval_l1(c_true-u_ref, Nx, Ny, Phi)
# l1_tau_threshold = 0.01 * l1_tau

iterMax_CFP = 50000
eta_CFP = 0.9

# Initialization: optimization coef
m = 5
eta = 0.5
iterMax = 50
wwp_c1 = 1e-5
wwp_c2 = 0.95

# Optimization main function
# u1, obj_fn = l_BFGS_CFP_box(u0, eval_fn_handle, eval_grad_handle, box_min, box_max, box_threshold; m=m, iterMax=iterMax, eta=eta, wwp_c1=wwp_c1, wwp_c2=wwp_c2, LinesearchMax=5, iterMax_CFP=iterMax_CFP, eta_CFP=eta_CFP)
u1, obj_fn = l_BFGS_CFP_box_tv(u0, eval_fn_handle, eval_grad_handle, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold; m=m, iterMax=iterMax, eta=eta, wwp_c1=wwp_c1, wwp_c2=wwp_c2, LinesearchMax=5, iterMax_CFP=iterMax_CFP, eta_CFP=eta_CFP)
# u1, obj_fn = l_BFGS_CFP_box_tv_avg(u0, eval_fn_handle, eval_grad_handle, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold,  proj_avg_handle, avg_tau, avg_tau_threshold; m=m, iterMax=iterMax, eta=eta, wwp_c1=wwp_c1, wwp_c2=wwp_c2, LinesearchMax=10, iterMax_CFP=iterMax_CFP, eta_CFP=eta_CFP)

# u1, obj_fn = l_BFGS_CFP_box_tv_l1(u0, eval_fn_handle, eval_grad_handle, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, Phi, u_ref, l1_tau, l1_tau_threshold; m=m, iterMax=iterMax, eta=eta, wwp_c1=wwp_c1, wwp_c2=wwp_c2, LinesearchMax=5, iterMax_CFP=iterMax_CFP, eta_CFP=eta_CFP)


# Saving
@printf "\nSaving...\n"
@save "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_result.jld2" u1 obj_fn
# @save "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_result_box_tv_l1.jld2" u1 obj_fn
@printf "Saved.\n"