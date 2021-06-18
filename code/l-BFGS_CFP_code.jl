using LinearAlgebra, JLD2, Printf
using Distributed, SharedArrays

# function handle is needed
# For example:
# eval_fn_handle(x) = eval_obj_fn(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
# eval_grad_handle(x) = compute_gradient(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
# proj_box_handle(x) = proj_box(x, Nx, Ny, minimum(c_true), maximum(c_true))
# proj_avg_handle(x) = proj_avg(x, Nx, Ny, position_ind, avg)

include("acoustic_solver.jl")
include("acoustic_solver_parallel.jl")
include("adjoint_method.jl")
include("optimization.jl")
include("projection_fn.jl")
include("total_variation.jl")
include("CFP.jl")

function l_BFGS_CFP_box(u0, eval_fn_handle, eval_grad_handle, box_min, box_max, box_threshold; m=5, iterMax=10, eta=0.5, wwp_c1=1e-15, wwp_c2=0.9, LinesearchMax=5, iterMax_CFP=100, eta_CFP=0.9)

    # Initialization: optimization
    # box_min_threshold = copy(box_threshold)
    # box_max_threshold = copy(box_threshold)
    
    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch update step eta = %f, max linesearch time = %d\n" eta LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # @printf "    Max CFP iteration time = %d, CFP_eta = %f\n" iterMax_CFP CFP_eta
    @printf "    box_min = %f, box_max = %f, box_threshold = %f\n" box_min box_max box_threshold
    # save name
    save_file_name0 = "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_temp_box_"
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
    lbfgs_count = 0
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

    grad0, fn_u0  = eval_grad_handle(u0)
    grad0 = grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, 0.1; alpha=1, maxSearch=LinesearchMax)
    grad1, fn_u1 = eval_grad_handle(u1)
    grad1 = grad1[:]
    # save
    obj_fn[iter] = fn_u0
    save_file_name = save_file_name0 * string(iter) * ".jld2"
    @save save_file_name u1 grad1

    for iter = 2:iterMax
        lbfgs_count += 1
        if lbfgs_count > m
            lbfgs_count = copy(m)
        end
        # update CFP threshold: linear
        if iter > 2
            # if minimum(u1) < box_min
            #     box_min -= box_min_threshold
            #     box_min_threshold = box_min_threshold * eta_CFP
            # end
            # if maximum(u1) > box_max
            #     box_max += box_max_threshold
            #     box_max_threshold = box_max_threshold * eta_CFP
            # end
            if minimum(u1) < box_min || maximum(u1) > box_max
                box_min -= box_threshold
                box_max += box_threshold
                box_threshold = box_threshold * eta_CFP
            end
        end
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count
        @printf "CFP threshold: box_min = %f, box_max = %f, box_threshold = %f\n" box_min box_max box_threshold

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
        u1_bar = quad_CFP_box(u1_tilde, u1, Nx, Ny, box_min, box_max, box_threshold, S[:,end-lbfgs_count+1:end], Y[:,end-lbfgs_count+1:end], R[end-lbfgs_count+1:end,end-lbfgs_count+1:end], D[end-lbfgs_count+1:end,end-lbfgs_count+1:end], L[end-lbfgs_count+1:end,end-lbfgs_count+1:end]; iterMax_CFP=iterMax_CFP)
        u2 = WWP_linesearch(u1_bar, u1, fn_u1, grad1, eval_grad_handle, eta; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=1, maxSearch=LinesearchMax)
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

function l_BFGS_CFP_box_tv(u0, eval_fn_handle, eval_grad_handle, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold; m=5, iterMax=10, eta=0.5, wwp_c1=1e-15, wwp_c2=0.9, LinesearchMax=5, iterMax_CFP=100, eta_CFP=0.9)

    # Initialization: optimization
    # box_min_threshold = copy(box_threshold)
    # box_max_threshold = copy(box_threshold)

    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch update step eta = %f, max linesearch time = %d\n" eta LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # @printf "    Max CFP iteration time = %d, CFP_eta = %f\n" iterMax_CFP CFP_eta
    @printf "    tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold
    # save name
    save_file_name0 = "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_temp_box_tv_"
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
    lbfgs_count = 0
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

    grad0, fn_u0  = eval_grad_handle(u0)
    grad0 = grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, 0.1; alpha=1, maxSearch=LinesearchMax)
    grad1, fn_u1 = eval_grad_handle(u1)
    grad1 = grad1[:]
    # save
    obj_fn[iter] = fn_u0
    save_file_name = save_file_name0 * string(iter) * ".jld2"
    @save save_file_name u1 grad1

    for iter = 2:iterMax
        lbfgs_count += 1
        if lbfgs_count > m
            lbfgs_count = copy(m)
        end
        # update CFP threshold: linear
        if iter > 2
            u1_tv = eval_tv(u1, Nx, Ny)
            if u1_tv > tv_tau
                tv_tau += tv_tau_threshold
                tv_tau_threshold = tv_tau_threshold * eta_CFP
            end
            # if minimum(u1) < box_min
            #     box_min -= box_min_threshold
            #     box_min_threshold = box_min_threshold * eta_CFP
            # end
            # if maximum(u1) > box_max
            #     box_max += box_max_threshold
            #     box_max_threshold = box_max_threshold * eta_CFP
            # end
            if minimum(u1) < box_min || maximum(u1) > box_max
                box_min -= box_threshold
                box_max += box_threshold
                box_threshold = box_threshold * eta_CFP
            end
        end
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count
        @printf "CFP threshold: tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold

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
        u1_bar = quad_CFP_box_tv(u1_tilde, u1, Nx, Ny, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, S[:,end-lbfgs_count+1:end], Y[:,end-lbfgs_count+1:end], R[end-lbfgs_count+1:end,end-lbfgs_count+1:end], D[end-lbfgs_count+1:end,end-lbfgs_count+1:end], L[end-lbfgs_count+1:end,end-lbfgs_count+1:end]; iterMax_CFP=iterMax_CFP)
        u2 = WWP_linesearch(u1_bar, u1, fn_u1, grad1, eval_grad_handle, eta; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=1, maxSearch=LinesearchMax)
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

function l_BFGS_CFP_box_tv_avg(u0, eval_fn_handle, eval_grad_handle, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, proj_avg_handle, avg_tau, avg_tau_threshold; m=5, iterMax=10, eta=0.5, wwp_c1=1e-15, wwp_c2=0.9, LinesearchMax=5, iterMax_CFP=100, eta_CFP=0.9)

    # Initialization: optimization
    # box_min_threshold = copy(box_threshold)
    # box_max_threshold = copy(box_threshold)

    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch update step eta = %f, max linesearch time = %d\n" eta LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # @printf "    Max CFP iteration time = %d, CFP_eta = %f\n" iterMax_CFP CFP_eta
    @printf "    tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f, avg_tau_threshold = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold avg_tau_threshold
    # save name
    save_file_name0 = "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_temp_box_tv_avg_"
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
    lbfgs_count = 0
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

    grad0, fn_u0  = eval_grad_handle(u0)
    grad0 = grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, 0.1; alpha=1, maxSearch=LinesearchMax)
    grad1, fn_u1 = eval_grad_handle(u1)
    grad1 = grad1[:]
    # save
    obj_fn[iter] = fn_u0
    save_file_name = save_file_name0 * string(iter) * ".jld2"
    @save save_file_name u1 grad1

    for iter = 2:iterMax
        lbfgs_count += 1
        if lbfgs_count > m
            lbfgs_count = copy(m)
        end
        if iter > 2
            u1_tv = eval_tv(u1, Nx, Ny)
            if u1_tv > tv_tau
                tv_tau += tv_tau_threshold
                tv_tau_threshold = tv_tau_threshold * eta_CFP
            end
            # if minimum(u1) < box_min
            #     box_min -= box_min_threshold
            #     box_min_threshold = box_min_threshold * eta_CFP
            # end
            # if maximum(u1) > box_max
            #     box_max += box_max_threshold
            #     box_max_threshold = box_max_threshold * eta_CFP
            # end
            if minimum(u1) < box_min || maximum(u1) > box_max
                box_min -= box_threshold
                box_max += box_threshold
                box_threshold = box_threshold * eta_CFP
            end
            p_avg = proj_avg_handle(u0)
            p_avg = p_avg[:]
            if norm(p_avg-u1) > avg_tau
                avg_tau += avg_tau_threshold
                avg_tau_threshold = avg_tau_threshold * eta_CFP
            end
        end
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count
        @printf "CFP threshold: tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f, avg_tau_threshold = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold avg_tau_threshold

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
        u1_bar = quad_CFP_box_tv_avg(u1_tilde, u1, Nx, Ny, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, proj_avg_handle, avg_tau, S[:,end-lbfgs_count+1:end], Y[:,end-lbfgs_count+1:end], R[end-lbfgs_count+1:end,end-lbfgs_count+1:end], D[end-lbfgs_count+1:end,end-lbfgs_count+1:end], L[end-lbfgs_count+1:end,end-lbfgs_count+1:end]; iterMax_CFP=iterMax_CFP)
        u2 = WWP_linesearch(u1_bar, u1, fn_u1, grad1, eval_grad_handle, eta; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=1, maxSearch=LinesearchMax)
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

function l_BFGS_CFP_box_tv_avg2(u0, eval_fn_handle, eval_grad_handle, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, proj_avg_handle1, proj_avg_handle2, avg_tau1, avg_tau2, avg_tau_threshold1, avg_tau_threshold2; m=5, iterMax=10, eta=0.5, wwp_c1=1e-15, wwp_c2=0.9, LinesearchMax=5, iterMax_CFP=100, eta_CFP=0.9)

    # Initialization: optimization
    # box_min_threshold = copy(box_threshold)
    # box_max_threshold = copy(box_threshold)

    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch update step eta = %f, max linesearch time = %d\n" eta LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # @printf "    Max CFP iteration time = %d, CFP_eta = %f\n" iterMax_CFP CFP_eta
    @printf "    tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f, avg_tau_threshold = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold avg_tau_threshold
    # save name
    save_file_name0 = "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_temp_box_tv_avg_"
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
    lbfgs_count = 0
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

    grad0, fn_u0  = eval_grad_handle(u0)
    grad0 = grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, 0.1; alpha=1, maxSearch=LinesearchMax)
    grad1, fn_u1 = eval_grad_handle(u1)
    grad1 = grad1[:]
    # save
    obj_fn[iter] = fn_u0
    save_file_name = save_file_name0 * string(iter) * ".jld2"
    @save save_file_name u1 grad1

    for iter = 2:iterMax
        lbfgs_count += 1
        if lbfgs_count > m
            lbfgs_count = copy(m)
        end
        if iter > 2
            u1_tv = eval_tv(u1, Nx, Ny)
            if u1_tv > tv_tau
                tv_tau += tv_tau_threshold
                tv_tau_threshold = tv_tau_threshold * eta_CFP
            end
            # if minimum(u1) < box_min
            #     box_min -= box_min_threshold
            #     box_min_threshold = box_min_threshold * eta_CFP
            # end
            # if maximum(u1) > box_max
            #     box_max += box_max_threshold
            #     box_max_threshold = box_max_threshold * eta_CFP
            # end
            if minimum(u1) < box_min || maximum(u1) > box_max
                box_min -= box_threshold
                box_max += box_threshold
                box_threshold = box_threshold * eta_CFP
            end
            p_avg1 = proj_avg_handle1(u0)
            p_avg1 = p_avg1[:]
            if norm(p_avg1-u1) > avg_tau1
                avg_tau1 += avg_tau_threshold1
                avg_tau_threshold1 = avg_tau_threshold1 * eta_CFP
            end
            p_avg2 = proj_avg_handle2(u0)
            p_avg2 = p_avg2[:]
            if norm(p_avg2-u1) > avg_tau2
                avg_tau2 += avg_tau_threshold2
                avg_tau_threshold2 = avg_tau_threshold2 * eta_CFP
            end
        end
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count
        @printf "CFP threshold: tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f, avg_tau_threshold1 = %f, avg_tau_threshold2 = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold avg_tau_threshold1 avg_tau_threshold2

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
        u1_bar = quad_CFP_box_tv_avg2(u1_tilde, u1, Nx, Ny, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, proj_avg_handle1, avg_tau1, proj_avg_handle2, avg_tau2, S[:,end-lbfgs_count+1:end], Y[:,end-lbfgs_count+1:end], R[end-lbfgs_count+1:end,end-lbfgs_count+1:end], D[end-lbfgs_count+1:end,end-lbfgs_count+1:end], L[end-lbfgs_count+1:end,end-lbfgs_count+1:end]; iterMax_CFP=iterMax_CFP)
        u2 = WWP_linesearch(u1_bar, u1, fn_u1, grad1, eval_grad_handle, eta; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=1, maxSearch=LinesearchMax)
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

function l_BFGS_CFP_box_tv_l1(u0, eval_fn_handle, eval_grad_handle, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, Phi, u_ref, l1_tau, l1_tau_threshold; m=5, iterMax=10, eta=0.5, wwp_c1=1e-15, wwp_c2=0.9, LinesearchMax=5, iterMax_CFP=100, eta_CFP=0.9)

    # Initialization: optimization
    # box_min_threshold = copy(box_threshold)
    # box_max_threshold = copy(box_threshold)

    u0 = u0[:]
    u_ref = u_ref[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch update step eta = %f, max linesearch time = %d\n" eta LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # @printf "    Max CFP iteration time = %d, CFP_eta = %f\n" iterMax_CFP CFP_eta
    @printf "    tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f, l1_tau = %f, l1_tau_threshold = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold l1_tau l1_tau_threshold
    # save name
    save_file_name0 = "/Users/lida/Desktop/Proj_grad_method/temp_data/lbfgs_CFP_temp_box_tv_l1_"
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
    lbfgs_count = 0
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

    grad0, fn_u0  = eval_grad_handle(u0)
    grad0 = grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, 0.1; alpha=1, maxSearch=LinesearchMax)
    grad1, fn_u1 = eval_grad_handle(u1)
    grad1 = grad1[:]
    # save
    obj_fn[iter] = fn_u0
    save_file_name = save_file_name0 * string(iter) * ".jld2"
    @save save_file_name u1 grad1

    for iter = 2:iterMax
        lbfgs_count += 1
        if lbfgs_count > m
            lbfgs_count = copy(m)
        end
        if iter > 2
            u1_tv = eval_tv(u1, Nx, Ny)
            if u1_tv > tv_tau
                tv_tau += tv_tau_threshold
                tv_tau_threshold = tv_tau_threshold * eta_CFP
            end
            # if minimum(u1) < box_min
            #     box_min -= box_min_threshold
            #     box_min_threshold = box_min_threshold * eta_CFP
            # end
            # if maximum(u1) > box_max
            #     box_max += box_max_threshold
            #     box_max_threshold = box_max_threshold * eta_CFP
            # end
            if minimum(u1) < box_min || maximum(u1) > box_max
                box_min -= box_threshold
                box_max += box_threshold
                box_threshold = box_threshold * eta_CFP
            end
            u1_l1 = eval_l1(u1-u_ref, Nx, Ny, Phi)
            if u1_l1 > l1_tau
                l1_tau += l1_tau_threshold
                l1_tau_threshold = l1_tau_threshold * eta_CFP
            end
        end
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count
        @printf "CFP threshold: tv_tau = %f, tv_tau_threshold = %f, box_min = %f, box_max = %f, box_threshold = %f, l1_tau = %f, l1_tau_threshold = %f\n" tv_tau tv_tau_threshold box_min box_max box_threshold l1_tau l1_tau_threshold

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
        u1_bar = quad_CFP_box_tv_l1(u1_tilde, u1, Nx, Ny, tv_tau, tv_tau_threshold, box_min, box_max, box_threshold, Phi, u_ref, l1_tau, l1_tau_threshold, S[:,end-lbfgs_count+1:end], Y[:,end-lbfgs_count+1:end], R[end-lbfgs_count+1:end,end-lbfgs_count+1:end], D[end-lbfgs_count+1:end,end-lbfgs_count+1:end], L[end-lbfgs_count+1:end,end-lbfgs_count+1:end]; iterMax_CFP=iterMax_CFP)
        u2 = WWP_linesearch(u1_bar, u1, fn_u1, grad1, eval_grad_handle, eta; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=1, maxSearch=LinesearchMax)
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