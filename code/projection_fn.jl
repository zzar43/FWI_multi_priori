using LinearAlgebra

# projection function of box constraint
function proj_box(u, Nx, Ny, a, b)
    u = reshape(u, Nx, Ny)
    proj_box_u = copy(u)
    proj_box_u[findall(x->x<a, u)] .= a
    proj_box_u[findall(x->x>b, u)] .= b
    return proj_box_u
end

function proj_box_mc(u, Nx, Ny, Nc, a, b)
    u = reshape(u, Nx, Ny, Nc)
    uu = zeros(Nx, Ny)
    for i = 1:Nc
        uu = copy(u[:,:,i])
        uu[findall(x->x<a[i], uu)] .= a[i]
        uu[findall(x->x>b[i], uu)] .= b[i]
        u[:,:,i] = uu
    end
    u = u[:]
    return u
end

# projection function of average constraint
# example: 
# position_ind = zeros(Nx, Ny)
# position_ind[5:7, 6:9] .= 1
# avg = -19

# function proj_avg(u, Nx, Ny, position_ind, avg)
#     u = reshape(u, Nx, Ny)
#     position_ind = reshape(position_ind, Nx, Ny)
#     proj_avg_u = u + (avg*norm(position_ind,2)^2 - sum(u.*position_ind))/norm(position_ind,2)^2 * position_ind
#     return proj_avg_u
# end

function proj_avg_mc(u, Nx, Ny, Nc, position_ind, avg)
    u = reshape(u, Nx, Ny, Nc)
    position_ind = reshape(position_ind, Nx, Ny, Nc)
    proj_avg_u = 0 .* u
    for i = 1:Nc
        proj_avg_u[:,:,i] = u[:,:,i] + (avg[i]*norm(position_ind[:,:,i],2)^2 - sum(u[:,:,i].*position_ind[:,:,i]))/norm(position_ind[:,:,i],2)^2 * position_ind[:,:,i]
    end
    proj_avg_u = proj_avg_u[:]
    return proj_avg_u
end

# update: 20200916
function proj_avg(u, Nx, Ny, position_ind, avg; N_avg=1)
    u = reshape(u, Nx, Ny)
    position_ind = reshape(position_ind, Nx, Ny, N_avg)
    proj_avg_u = zeros(Nx, Ny, N_avg)
    for i = 1:N_avg
        position_ind0 = position_ind[:,:,i]
        proj_avg_u[:,:,i] = u + (avg[i]*norm(position_ind0,2)^2 - sum(u.*position_ind0))/norm(position_ind0,2)^2 * position_ind0
    end
    proj_avg_u0 = sum(proj_avg_u, dims=3)
    proj_avg_u0 = proj_avg_u0[:,:,1]
    proj_avg_u0 = proj_avg_u0 ./ N_avg
    return proj_avg_u0
end

function proj_subspace(u, position_ind, sub_vector)
    ind = findall(x->x==1,position_ind)
    u[ind] .= sub_vector[ind]
    return u
end

# Subgradient projection of l1 ball
function eval_l1(u, Nx, Ny, Phi)
    # u = reshape(u, Nx*Ny, 1)
    # g_u = 0
    # for j = 1:size(Phi,2)
    #     g_u += abs(sum(Phi[:,j] .* u))
    # end
    # return g_u

    # return norm(Phi' * u[:], 1)

    return norm(u[:], 1)
end

function subgrad_l1(u, Nx, Ny, Phi)
    # u = reshape(u, Nx*Ny, 1)
    # subgrad_l1_u = zeros(Nx*Ny,1)
    # Phi_u = Phi' * u
    # for i = 1:Nx*Ny
    #     for j = 1:size(Phi,2)
    #         subgrad_l1_u[i] += sign(Phi_u[j]) * Phi[i,j]
    #     end
    # end
    # subgrad_l1_u = reshape(subgrad_l1_u, Nx, Ny)

    subgrad_l1_u = sign.(u)
    return subgrad_l1_u
end