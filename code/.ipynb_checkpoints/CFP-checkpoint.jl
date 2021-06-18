using Printf
# CFP 1
# text example. 3 constraints: TV, Box, Avg
# define function handle:
# position_ind = zeros(Nx, Ny)
# position_ind[1:16,:] .= 1
# avg = 1.5
# proj_box_handle(x) = proj_box(x, Nx, Ny, a, b)
# proj_avg_handle(x) = proj_avg(x, Nx, Ny, position_ind, avg)

function CFP(u, Nx, Ny, tv_tau, proj_box_handle, proj_avg_handle; iterMax=50, eps=0.1)
    x0 = copy(u)
    x1 = zeros(Nx, Ny)
    p_tv = zeros(Nx, Ny)
    p_box = zeros(Nx, Ny)
    p_avg = zeros(Nx, Ny)
    
    for iter = 1:iterMax
        
        tv_x0 = eval_tv(x0, Nx, Ny)
        if tv_x0 <= (1+eps)*tv_tau
            print("Iteration: ")
            print(iter)
            println(". tv(x_0) <= (1+eps)*tv_tau,  converge.")
            break
        end
        
        # projection
        if tv_x0 > tv_tau
            subgrad_tv_x0 = subgrad_tv(x0, Nx, Ny)
            p_tv = x0 - (tv_x0-tv_tau)/norm(subgrad_tv_x0,2)^2 * subgrad_tv_x0
        else
            p_tv = x0
        end
        p_box = proj_box_handle(x0)
        p_avg = proj_avg_handle(x0)

        if tv_x0 > tv_tau
            lambda = 1/3*norm(p_tv-x0,2)^2 + 1/3*norm(p_box-x0,2)^2 + 1/3*norm(p_avg-x0,2)^2
            lambda = lambda / norm(x0 - 1/3 .* p_tv - 1/3 .* p_box - 1/3 .* p_avg, 2)^2
        else
            lambda = 1
        end
        x1 = x0 - lambda .* (x0 - 1/3 .* p_tv - 1/3 .* p_box - 1/3 .* p_avg)
        x0 = copy(x1)
    end
    return x1
end

# proj_box_handle(x) = proj_box_mc(x, Nx, Ny, Nc, a, b)
# proj_avg_handle(x) = proj_avg_mc(x, Nx, Ny, Nc, position_ind, avg)
function CFP_mc(u, Nx, Ny, Nc, tv_tau, proj_box_handle, proj_avg_handle; iterMax=50, eps=0.1)
    x0 = copy(u)
    x1 = zeros(Nx, Ny, Nc)
    p_tv = zeros(Nx, Ny, Nc)
    p_box = zeros(Nx, Ny, Nc)
    p_avg = zeros(Nx, Ny, Nc)
    
    for iter = 1:iterMax
        
        tv_x0 = eval_vtv(x0, Nx, Ny, Nc)
        if tv_x0 <= (1+eps)*tv_tau
            print("Iteration: ")
            print(iter)
            println(". vtv(x_0) <= (1+eps)*tv_tau,  converge.")
            break
        end
        
        # projection
        if tv_x0 > tv_tau
            subgrad_tv_x0 = subgrad_vtv(x0, Nx, Ny, Nc)
            p_tv = x0 - (tv_x0-tv_tau)/norm(subgrad_tv_x0,2)^2 * subgrad_tv_x0
        else
            p_tv = x0
        end
        p_box = proj_box_handle(x0)
        p_avg = proj_avg_handle(x0)

        if tv_x0 > tv_tau
            lambda = 1/3*norm(p_tv-x0,2)^2 + 1/3*norm(p_box-x0,2)^2 + 1/3*norm(p_avg-x0,2)^2
            lambda = lambda / norm(x0 - 1/3 .* p_tv - 1/3 .* p_box - 1/3 .* p_avg, 2)^2
        else
            lambda = 1
        end
        x1 = x0 - lambda .* (x0 - 1/3 .* p_tv - 1/3 .* p_box - 1/3 .* p_avg)
        x0 = copy(x1)
    end
    return x0
end

# Quad CFP 1: box, avg, tv
# initialization is needed, for example:
# tv_tau = 0.5 * eval_tv(c_true, Nx, Ny)
# position_ind = zeros(Nx, Ny)
# position_ind[1:16,:] .= 1
# avg = sum(position_ind[:] .* c_true) ./ norm(position_ind,2)^2
# proj_box_handle(x) = proj_box(x, Nx, Ny, minimum(c_true), maximum(c_true))
# proj_avg_handle(x) = proj_avg(x, Nx, Ny, position_ind, avg)

function quad_CFP_1(u0, Nx, Ny, tv_tau, proj_box_handle, proj_avg_handle, S, Y, R, D, L; iterMax_CFP=100, eps_tv=0.01, eps_box=0.01, eps_avg=0.01)
    
    u0 = u0[:]
    u0_backup = copy(u0)
    u1 = 0 * u0;

    s0 = S[:,end]
    y0 = Y[:,end]
    
    for i = 1:(iterMax_CFP+1)
        if i == (iterMax_CFP+1)
            u0_tv = eval_tv(u0, Nx, Ny)
            p_box = proj_box_handle(u0)
            a_box = p_box[:] - u0
            p_avg = proj_avg_handle(u0)
            a_avg = p_avg[:] - u0
            @printf "    Distance to TV constraint set: |u0|_tv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Distance to average constraint set: %f\n" norm(a_avg,2)
            @printf "    Subproblem iteration: %d, not converged.\n" iterMax_CFP
            break;
        end

        # 1
        g_tv = eval_tv(u0, Nx, Ny)
        # projection
        if g_tv > tv_tau
            t_tv = subgrad_tv(u0, Nx, Ny)
            t_tv = t_tv[:]
            p_tv = u0 - (g_tv-tv_tau)/norm(t_tv,2)^2 * t_tv
        else
            p_tv = u0
        end
        a_tv = p_tv - u0
        p_box = proj_box_handle(u0)
        a_box = p_box[:] - u0
        p_avg = proj_avg_handle(u0)
        a_avg = p_avg[:] - u0
        # check if break
        u0_tv = eval_tv(u0, Nx, Ny)
        if (u0_tv <= eps_tv+tv_tau) && (norm(a_box,2) <= eps_box) && (norm(a_avg,2)<= eps_avg)
            @printf "    Distance to TV constraint set: |u0|_tv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Distance to average constraint set: %f\n" norm(a_avg,2)
            @printf "    Subproblem iteration: %d, converged.\n" i
            break;
        end

        # 2
        v = (a_tv + a_box + a_avg) ./ 3
        v = v[:]
        # print tv of projection point
        # g_tv_v = eval_tv(u0+v, Nx, Ny)
        # if (norm(v) < epsilon) && i<=iterMax_CFP
        #     @printf "    tv(proj_u0) = %1.5e, tv_tau = %1.5e\n" g_tv_v tv_tau
        #     @printf "    CFP iteration: %d, v = (a_tv + a_box + a_avg) ./ 3 < epsilon, break.\n" i
        #     break;
        # elseif (norm(v) >= epsilon) && i==iterMax_CFP
        #     @printf "    tv(proj_u0) = %1.5e, tv_tau = %1.5e\n" g_tv_v tv_tau
        #     @printf "    CFP iteration: %d, v = (a_tv + a_box + a_avg) ./ 3 > epsilon, end.\n" i
        # end
        lambda = (norm(a_tv,2)^2 + norm(a_box,2)^2 + norm(a_avg,2)^2) ./ 3

        # 3
        if lambda == 0
            break;
        end
        b = u0_backup - u0
        if maximum(abs.(b)) == 0
            c = zeros(Nx * Ny, 1)
        else
            c = compute_B_ku(b, S, Y, L, D)
        end
        d = compute_H_ku(v, S, Y, R, D)
        lambda = lambda / sum(d' * v)[1]

        # 4
        d = lambda * d

        # 5
        pii = -sum(c'*d)[1]
        mu = sum(b'*c)[1]
        nu = lambda * sum(d'*v)[1]
        rho = mu*nu - pii^2

        # 6
        if rho == 0 && pii >= 0
            u1 = u0 + d
        elseif rho > 0 && pii*nu >= rho
            u1 = u0_backup + (1+pii/nu)*d
        elseif rho > 0 && pii*nu < rho
            u1 = u0 + nu/rho * (pii*b + mu*d)
        else
            println("error rho < 0")
            break
        end

        u0 = copy(u1)
    end
    return u0
end


function quad_CFP_box(u0, Nx, Ny, tv_tau, proj_box_handle, proj_avg_handle, S, Y, R, D, L; iterMax_CFP=100, eps_tv=0.01, eps_box=0.01, eps_avg=0.01)
    
    u0 = u0[:]
    u0_backup = copy(u0)
    u1 = 0 * u0;

    s0 = S[:,end]
    y0 = Y[:,end]
    
    for i = 1:(iterMax_CFP+1)
        if i == (iterMax_CFP+1)
            p_box = proj_box_handle(u0)
            a_box = p_box[:] - u0
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Subproblem iteration: %d, not converged.\n" iterMax_CFP
            break;
        end

        # 1
        # projection
        p_box = proj_box_handle(u0)
        a_box = p_box[:] - u0
        # check if break
        if (norm(a_box,2) <= eps_box)
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Subproblem iteration: %d, converged.\n" i
            break;
        end

        # 2
        v = (a_box) ./ 1
        v = v[:]
        lambda = (norm(a_box,2)^2) ./ 1

        # 3
        if lambda == 0
            break;
        end
        b = u0_backup - u0
        if maximum(abs.(b)) == 0
            c = zeros(Nx * Ny, 1)
        else
            c = compute_B_ku(b, S, Y, L, D)
        end
        d = compute_H_ku(v, S, Y, R, D)
        lambda = lambda / sum(d' * v)[1]

        # 4
        d = lambda * d

        # 5
        pii = -sum(c'*d)[1]
        mu = sum(b'*c)[1]
        nu = lambda * sum(d'*v)[1]
        rho = mu*nu - pii^2

        # 6
        if rho == 0 && pii >= 0
            u1 = u0 + d
        elseif rho > 0 && pii*nu >= rho
            u1 = u0_backup + (1+pii/nu)*d
        elseif rho > 0 && pii*nu < rho
            u1 = u0 + nu/rho * (pii*b + mu*d)
        else
            println("error rho < 0")
            break
        end

        u0 = copy(u1)
    end
    return u0
end

function quad_CFP_avg(u0, Nx, Ny, tv_tau, proj_box_handle, proj_avg_handle, S, Y, R, D, L; iterMax_CFP=100, eps_tv=0.01, eps_box=0.01, eps_avg=0.01)
    
    u0 = u0[:]
    u0_backup = copy(u0)
    u1 = 0 * u0;

    s0 = S[:,end]
    y0 = Y[:,end]
    
    for i = 1:(iterMax_CFP+1)
        if i == (iterMax_CFP+1)
            p_avg = proj_avg_handle(u0)
            a_avg = p_avg[:] - u0
            @printf "    Distance to average constraint set: %f\n" norm(a_avg,2)
            @printf "    Subproblem iteration: %d, not converged.\n" iterMax_CFP
            break;
        end

        # 1
        p_avg = proj_avg_handle(u0)
        a_avg = p_avg[:] - u0
        # check if break
        if (norm(a_avg,2)<= eps_avg)
            @printf "    Distance to average constraint set: %f\n" norm(a_avg,2)
            @printf "    Subproblem iteration: %d, converged.\n" i
            break;
        end

        # 2
        v = (a_avg) ./ 1
        v = v[:]
        lambda = (norm(a_avg,2)^2) ./ 1

        # 3
        if lambda == 0
            break;
        end
        b = u0_backup - u0
        if maximum(abs.(b)) == 0
            c = zeros(Nx * Ny, 1)
        else
            c = compute_B_ku(b, S, Y, L, D)
        end
        d = compute_H_ku(v, S, Y, R, D)
        lambda = lambda / sum(d' * v)[1]

        # 4
        d = lambda * d

        # 5
        pii = -sum(c'*d)[1]
        mu = sum(b'*c)[1]
        nu = lambda * sum(d'*v)[1]
        rho = mu*nu - pii^2

        # 6
        if rho == 0 && pii >= 0
            u1 = u0 + d
        elseif rho > 0 && pii*nu >= rho
            u1 = u0_backup + (1+pii/nu)*d
        elseif rho > 0 && pii*nu < rho
            u1 = u0 + nu/rho * (pii*b + mu*d)
        else
            println("error rho < 0")
            break
        end

        u0 = copy(u1)
    end
    return u0
end

function quad_CFP_tv(u0, Nx, Ny, tv_tau, proj_box_handle, proj_avg_handle, S, Y, R, D, L; iterMax_CFP=100, eps_tv=0.01, eps_box=0.01, eps_avg=0.01)
    
    u0 = u0[:]
    u0_backup = copy(u0)
    u1 = 0 * u0;

    s0 = S[:,end]
    y0 = Y[:,end]
    
    for i = 1:(iterMax_CFP+1)
        if i == (iterMax_CFP+1)
            u0_tv = eval_tv(u0, Nx, Ny)
            @printf "    Distance to TV constraint set: |u0|_tv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Subproblem iteration: %d, not converged.\n" iterMax_CFP
            break;
        end

        # 1
        g_tv = eval_tv(u0, Nx, Ny)
        # projection
        if g_tv > tv_tau
            t_tv = subgrad_tv(u0, Nx, Ny)
            t_tv = t_tv[:]
            p_tv = u0 - (g_tv-tv_tau)/norm(t_tv,2)^2 * t_tv
        else
            p_tv = u0
        end
        a_tv = p_tv - u0
        # check if break
        u0_tv = eval_tv(u0, Nx, Ny)
        if (u0_tv <= eps_tv+tv_tau)
            @printf "    Distance to TV constraint set: |u0|_tv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Subproblem iteration: %d, converged.\n" i
            break;
        end

        # 2
        v = (a_tv) ./ 1
        v = v[:]
        lambda = (norm(a_tv,2)^2) ./ 1

        # 3
        if lambda == 0
            break;
        end
        b = u0_backup - u0
        if maximum(abs.(b)) == 0
            c = zeros(Nx * Ny, 1)
        else
            c = compute_B_ku(b, S, Y, L, D)
        end
        d = compute_H_ku(v, S, Y, R, D)
        lambda = lambda / sum(d' * v)[1]

        # 4
        d = lambda * d

        # 5
        pii = -sum(c'*d)[1]
        mu = sum(b'*c)[1]
        nu = lambda * sum(d'*v)[1]
        rho = mu*nu - pii^2

        # 6
        if rho == 0 && pii >= 0
            u1 = u0 + d
        elseif rho > 0 && pii*nu >= rho
            u1 = u0_backup + (1+pii/nu)*d
        elseif rho > 0 && pii*nu < rho
            u1 = u0 + nu/rho * (pii*b + mu*d)
        else
            println("error rho < 0")
            break
        end

        u0 = copy(u1)
    end
    return u0
end

function quad_CFP_box_tv(u0, Nx, Ny, tv_tau, proj_box_handle, proj_avg_handle, S, Y, R, D, L; iterMax_CFP=100, eps_tv=0.01, eps_box=0.01, eps_avg=0.01)

    u0 = u0[:]
    u0_backup = copy(u0)
    u1 = 0 * u0;

    s0 = S[:,end]
    y0 = Y[:,end]

    converge = false
    i = 0
    
    # for i = 1:(iterMax_CFP+1)
        # if i == (iterMax_CFP+1)
        #     u0_tv = eval_tv(u0, Nx, Ny)
        #     p_box = proj_box_handle(u0)
        #     a_box = p_box[:] - u0
        #     @printf "    Distance to TV constraint set: |u0|_tv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
        #     @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
        #     @printf "    Subproblem iteration: %d, not converged.\n" iterMax_CFP
        #     break;
        # end
    while converge == false
        i += 1
        # 1
        g_tv = eval_tv(u0, Nx, Ny)
        # projection
        if g_tv > tv_tau
            t_tv = subgrad_tv(u0, Nx, Ny)
            t_tv = t_tv[:]
            p_tv = u0 - (g_tv-tv_tau)/norm(t_tv,2)^2 * t_tv
        else
            p_tv = u0
        end
        a_tv = p_tv - u0
        p_box = proj_box_handle(u0)
        a_box = p_box[:] - u0
        # check if break
        u0_tv = eval_tv(u0, Nx, Ny)
        if (u0_tv <= eps_tv+tv_tau) && (norm(a_box,2) <= eps_box)
            @printf "    Distance to TV constraint set: |u0|_tv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Subproblem iteration: %d, converged.\n" i
            converge = true
            break;
        end

        # 2
        v = (a_tv + a_box) ./ 2
        v = v[:]
        lambda = (norm(a_tv,2)^2 + norm(a_box,2)^2) ./ 2

        # 3
        if lambda == 0
            break;
        end
        b = u0_backup - u0
        if maximum(abs.(b)) == 0
            c = zeros(Nx * Ny, 1)
        else
            c = compute_B_ku(b, S, Y, L, D)
        end
        d = compute_H_ku(v, S, Y, R, D)
        lambda = lambda / sum(d' * v)[1]

        # 4
        d = lambda * d

        # 5
        pii = -sum(c'*d)[1]
        mu = sum(b'*c)[1]
        nu = lambda * sum(d'*v)[1]
        rho = mu*nu - pii^2

        # 6
        if rho == 0 && pii >= 0
            u1 = u0 + d
        elseif rho > 0 && pii*nu >= rho
            u1 = u0_backup + (1+pii/nu)*d
        elseif rho > 0 && pii*nu < rho
            u1 = u0 + nu/rho * (pii*b + mu*d)
        else
            println("error rho < 0")
            break
        end

        u0 = copy(u1)
    end
    return u0
end




# ===================================
function quad_CFP_1_mc(u0, Nx, Ny, Nc, tv_tau, proj_box_handle, proj_avg_handle, S, Y, R, D, L; iterMax_CFP=100, eps_tv=0.01, eps_box=0.01, eps_avg=0.01)
    
    u0 = u0[:]
    u0_backup = copy(u0)
    u1 = 0 * u0;

    s0 = S[:,end]
    y0 = Y[:,end]
    
    for i = 1:(iterMax_CFP+1)
        if i == (iterMax_CFP+1)
            u0_tv = eval_vtv(u0, Nx, Ny, Nc)
            p_box = proj_box_handle(u0)
            a_box = p_box[:] - u0
            p_avg = proj_avg_handle(u0)
            a_avg = p_avg[:] - u0
            @printf "    Distance to VTV constraint set: |u0|_vtv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Distance to average constraint set: %f\n" norm(a_avg,2)
            @printf "    Subproblem iteration: %d, not converged.\n" iterMax_CFP
            break;
        end

        # 1
        g_tv = eval_vtv(u0, Nx, Ny, Nc)
        # projection
        if g_tv > tv_tau
            t_tv = subgrad_vtv(u0, Nx, Ny, Nc)
            t_tv = t_tv[:]
            p_tv = u0 - (g_tv-tv_tau)/norm(t_tv,2)^2 * t_tv
        else
            p_tv = u0
        end
        a_tv = p_tv - u0
        p_box = proj_box_handle(u0)
        a_box = p_box[:] - u0
        p_avg = proj_avg_handle(u0)
        a_avg = p_avg[:] - u0
        # check if break
        u0_tv = eval_vtv(u0, Nx, Ny, Nc)
        if (u0_tv <= eps_tv+tv_tau) && (norm(a_box,2) <= eps_box) && (norm(a_avg,2)<= eps_avg)
            @printf "    Distance to VTV constraint set: |u0|_vtv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Distance to average constraint set: %f\n" norm(a_avg,2)
            @printf "    Subproblem iteration: %d, converged.\n" i
            break;
        end

        # 2
        v = (a_tv + a_box + a_avg) ./ 3
        v = v[:]
        lambda = (norm(a_tv,2)^2 + norm(a_box,2)^2 + norm(a_avg,2)^2) ./ 3

        # 3
        if lambda == 0
            break;
        end
        b = u0_backup - u0
        if maximum(abs.(b)) == 0
            c = 0 * u0
        else
            c = compute_B_ku(b, S, Y, L, D)
        end
        d = compute_H_ku(v, S, Y, R, D)
        lambda = lambda / sum(d' * v)[1]

        # 4
        d = lambda * d

        # 5
        pii = -sum(c'*d)[1]
        mu = sum(b'*c)[1]
        nu = lambda * sum(d'*v)[1]
        rho = mu*nu - pii^2

        # 6
        if rho == 0 && pii >= 0
            u1 = u0 + d
        elseif rho > 0 && pii*nu >= rho
            u1 = u0_backup + (1+pii/nu)*d
        elseif rho > 0 && pii*nu < rho
            u1 = u0 + nu/rho * (pii*b + mu*d)
        else
            println("error rho < 0")
            break
        end

        u0 = copy(u1)
    end
    return u0
end

function quad_CFP_box_tv_mc(u0, Nx, Ny, Nc, tv_tau, proj_box_handle, proj_avg_handle, S, Y, R, D, L; iterMax_CFP=100, eps_tv=0.01, eps_box=0.01, eps_avg=0.01)
    
    u0 = u0[:]
    u0_backup = copy(u0)
    u1 = 0 * u0;

    s0 = S[:,end]
    y0 = Y[:,end]
    
    for i = 1:(iterMax_CFP+1)
        if i == (iterMax_CFP+1)
            u0_tv = eval_vtv(u0, Nx, Ny, Nc)
            p_box = proj_box_handle(u0)
            a_box = p_box[:] - u0
            @printf "    Distance to VTV constraint set: |u0|_vtv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Subproblem iteration: %d, not converged.\n" iterMax_CFP
            break;
        end

        # 1
        g_tv = eval_vtv(u0, Nx, Ny, Nc)
        # projection
        if g_tv > tv_tau
            t_tv = subgrad_vtv(u0, Nx, Ny, Nc)
            t_tv = t_tv[:]
            p_tv = u0 - (g_tv-tv_tau)/norm(t_tv,2)^2 * t_tv
        else
            p_tv = u0
        end
        a_tv = p_tv - u0
        p_box = proj_box_handle(u0)
        a_box = p_box[:] - u0
        # check if break
        u0_tv = eval_vtv(u0, Nx, Ny, Nc)
        if (u0_tv <= eps_tv+tv_tau) && (norm(a_box,2) <= eps_box) 
            @printf "    Distance to VTV constraint set: |u0|_vtv = %f, tau_tv + eps_tv = %f\n" u0_tv tv_tau+eps_tv
            @printf "    Distance to box constraint set: %f\n" norm(a_box,2)
            @printf "    Subproblem iteration: %d, converged.\n" i
            break;
        end

        # 2
        v = (a_tv + a_box) ./ 2
        v = v[:]
        lambda = (norm(a_tv,2)^2 + norm(a_box,2)^2) ./ 2

        # 3
        if lambda == 0
            break;
        end
        b = u0_backup - u0
        if maximum(abs.(b)) == 0
            c = 0 * u0
        else
            c = compute_B_ku(b, S, Y, L, D)
        end
        d = compute_H_ku(v, S, Y, R, D)
        lambda = lambda / sum(d' * v)[1]

        # 4
        d = lambda * d

        # 5
        pii = -sum(c'*d)[1]
        mu = sum(b'*c)[1]
        nu = lambda * sum(d'*v)[1]
        rho = mu*nu - pii^2

        # 6
        if rho == 0 && pii >= 0
            u1 = u0 + d
        elseif rho > 0 && pii*nu >= rho
            u1 = u0_backup + (1+pii/nu)*d
        elseif rho > 0 && pii*nu < rho
            u1 = u0 + nu/rho * (pii*b + mu*d)
        else
            println("error rho < 0")
            break
        end

        u0 = copy(u1)
    end
    return u0
end