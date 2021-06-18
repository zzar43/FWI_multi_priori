using PyPlot, LinearAlgebra, ImageFiltering, JLD2, MATLAB, MAT, Printf
using Distributed, SharedArrays

# function handle is needed
# For example:
# eval_fn_handle(x) = eval_obj_fn(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
# eval_grad_handle(x) = compute_gradient(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

include("acoustic_solver.jl")
include("acoustic_solver_parallel.jl")
include("adjoint_method.jl")
include("optimization.jl")

function l_BFGS(u0, eval_fn_handle, eval_grad_handle; m=5, iterMax=10, eta=0.9, LinesearchMax=5, wwp_c1=1e-4, wwp_c2=0.9)

    # Initialization: optimization
    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch update step eta = %f, max linesearch time = %d\n" eta LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # save name
    save_file_name0 = "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_temp_"
    obj_fn = zeros(iterMax)

    @printf "Preparing optimization...\n"
    # initialization
    S = zeros(n, m)
    Y = zeros(n, m)
    R = zeros(m, m)
    D = zeros(m, m)
    L = zeros(m, m)

    @printf "\nOptimization start.\n"
    # iter 1
    iter = 1
    lbfgs_count = min(iter-1,m)
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

    grad0, fn_u0  = eval_grad_handle(u0)
    grad0 = grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, eta; alpha=1, maxSearch=LinesearchMax)
    # u1 = WWP_linesearch(u_bar, u0, fn_u0, grad0, eval_grad_handle, eta; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=1, maxSearch=LinesearchMax)
    grad1, fn_u1 = eval_grad_handle(u1)
    grad1 = grad1[:]
    # save
    obj_fn[iter] = fn_u0
    save_file_name = save_file_name0 * string(iter) * ".jld2"
    @save save_file_name u1

    for iter = 2:iterMax
        lbfgs_count = min(iter-1,m)
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

        S[:,1:m-1] = S[:,2:m]
        S[:,m] = u1 - u0
        Y[:,1:m-1] = Y[:,2:m]
        Y[:,m] = grad1-grad0
        if sum(S[:,m].*Y[:,m]) < 0
            println("s^T y < 0, break.")
            break;
        end
        D[1:m-1,1:m-1] = D[2:m,2:m]
        D[m,m] = sum(S[:,m].*Y[:,m])
        R[1:m-1,1:m-1] = R[2:m,2:m]
        for i = 1:m
            R[i,m] = sum(S[:,i].*Y[:,m])
        end
        L[1:m-1,1:m-1] = L[2:m,2:m]
        for i = 1:m-1
            L[m,i] = sum(S[:,m].*Y[:,i])
        end
        u1_tilde = u1 - compute_H_ku(grad1, S[:,end-lbfgs_count+1:end], Y[:,end-lbfgs_count+1:end], R[end-lbfgs_count+1:end,end-lbfgs_count+1:end], D[end-lbfgs_count+1:end,end-lbfgs_count+1:end])
        # u2 = linesearch_back(u1_tilde, u1, fn_u1, eval_fn_handle, eta; alpha=1, maxSearch=LinesearchMax)
        u2 = WWP_linesearch(u1_tilde, u1, fn_u1, grad1, eval_grad_handle, eta; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=1, maxSearch=LinesearchMax)
        if u2 == u1
            println("Linesearch failed. Break.")
            break;
        end
        grad2, fn_u2 = eval_grad_handle(u2)
        grad2 = grad2[:]
        
        u0 = copy(u1)
        u1 = copy(u2)
        grad0 = copy(grad1)
        grad1 = copy(grad2)
        fn_u0 = copy(fn_u1)
        fn_u1 = copy(fn_u2)
        # save
        obj_fn[iter] = fn_u1
        save_file_name = save_file_name0 * string(iter) * ".jld2"
        @save save_file_name u1 grad1
    end
    return u1, obj_fn
end

function l_BFGS_mc(u0, eval_fn_handle, eval_grad_handle; m=5, iterMax=10, eta=0.9, LinesearchMax=5)

    # Initialization: optimization
    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch update step eta = %f, max linesearch time = %d\n" eta LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax

    @printf "Preparing optimization...\n"
    # initialization
    S = zeros(n, m)
    Y = zeros(n, m)
    R = zeros(m, m)
    D = zeros(m, m)
    L = zeros(m, m)

    @printf "\nOptimization start.\n"
    # iter 1
    iter = 1
    lbfgs_count = min(iter-1,m)
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

    grad0, fn_u0  = eval_grad_handle(u0)
    grad0= grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, eta; alpha=1, maxSearch=LinesearchMax)
    grad1, fn_u1 = eval_grad_handle(u1)
    grad1 = grad1[:]

    for iter = 2:iterMax
        lbfgs_count = min(iter-1,m)
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

        S[:,1:m-1] = S[:,2:m]
        S[:,m] = u1 - u0
        Y[:,1:m-1] = Y[:,2:m]
        Y[:,m] = grad1-grad0
        if sum(S[:,m].*Y[:,m]) < 0
            println("s^T y < 0, break.")
            break;
        end
        D[1:m-1,1:m-1] = D[2:m,2:m]
        D[m,m] = sum(S[:,m].*Y[:,m])
        R[1:m-1,1:m-1] = R[2:m,2:m]
        for i = 1:m
            R[i,m] = sum(S[:,i].*Y[:,m])
        end
        L[1:m-1,1:m-1] = L[2:m,2:m]
        for i = 1:m-1
            L[m,i] = sum(S[:,m].*Y[:,i])
        end
        u1_tilde = u1 - compute_H_ku(grad1, S[:,end-lbfgs_count+1:end], Y[:,end-lbfgs_count+1:end], R[end-lbfgs_count+1:end,end-lbfgs_count+1:end], D[end-lbfgs_count+1:end,end-lbfgs_count+1:end])
        u2 = linesearch_back(u1_tilde, u1, fn_u1, eval_fn_handle, eta; alpha=1, maxSearch=LinesearchMax)
        if u2 == u1
            println("Linesearch failed. Break.")
            break;
        end
        grad2, fn_u2 = eval_grad_handle(u2)
        grad2 = grad2[:]
        
        u0 = copy(u1)
        u1 = copy(u2)
        grad0 = copy(grad1)
        grad1 = copy(grad2)
        fn_u0 = copy(fn_u1)
        fn_u1 = copy(fn_u2)
    end
    return u1
end