using Distributed, LinearAlgebra
@everywhere include("code/acoustic_solver_parallel.jl")

# input: u
# output: y, Qy

function forward_modelling(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    data_forward = SharedArray{Float64}(Nt, receiver_num, source_num)
    wavefield_forward = SharedArray{Float64}(Nx, Ny, Nt, source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        U1, data1 = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        data_forward[:,:,ind] = data1;
        wavefield_forward[:,:,:,ind] = U1;
    end
    data_forward = Array(data_forward)
    wavefield_forward = Array(wavefield_forward)
    
    return wavefield_forward, data_forward
end


# (QDF[u])^* y_0 = u_0
# when y_0 = y_d - Qy, \nabla J(u) = u_0
# input: u, y, y_0
# output: u_0

function adjoint_op(wavefield_forward, y0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    u0 = SharedArray{Float64}(Nx, Ny, source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        @views adj_source = y0[end:-1:1,:,ind];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = zeros(Nx, Ny, Nt)
        @views @. utt[:,:,2:end-1] = (wavefield_forward[:,:,3:end,ind]-2*wavefield_forward[:,:,2:end-1,ind]+wavefield_forward[:,:,1:end-2,ind]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3) * dt
        u0[:,:,ind] = grad0
        
    end
    
    u0 = Array(u0)
    u0 = sum(u0, dims=3)
    u0 = u0[:,:,1]
    u0 = reshape(u0, Nx*Ny,1)
    
    return u0
end


# single scattering
function single_scattering(wavefield_forward, u0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    
    c = reshape(c, Nx, Ny)
    u0 = reshape(u0, Nx, Ny)
    source_num = size(wavefield_forward,4)
    receiver_num = size(receiver_position,1)
    data_single = SharedArray{Float64}(Nt, receiver_num, source_num)
    
    source_position_s = zeros(Int,Nx*Ny,2)
    for i = 1:Nx
        for j = 1:Ny
            source_position_s[(i-1)*Ny+j,1] = i
            source_position_s[(i-1)*Ny+j,2] = j
        end
    end
    
    @inbounds @sync @distributed for ind = 1:source_num
        
        # single scattering
        forward_wavefield_tt = zeros(Nx, Ny, Nt)
        @views @. forward_wavefield_tt[:,:,2:end-1] = (wavefield_forward[:,:,3:end,ind]-2*wavefield_forward[:,:,2:end-1,ind]+wavefield_forward[:,:,1:end-2,ind]) ./ dt^2;
        source_s = zeros(Nt, Nx*Ny)
        for i = 1:Nx
            for j = 1:Ny
                @. source_s[:,(i-1)*Ny+j] = 2*forward_wavefield_tt[i,j,:] .* u0[i,j] ./ (c[i,j]^3)
            end
        end
        data_single[:,:,ind] = acoustic_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source_s, source_position_s, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    end
    
    data_single = Array(data_single)
    
    return data_single
end

# normal operator for Gauss Newton method: (QDF[u])^* QDF[u] u_0 = u_1
# input: u, y, u_0
# output: u_1

function normal_op(wavefield_forward, u0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    
    c = reshape(c, Nx, Ny)
    u0 = reshape(u0, Nx, Ny)
    source_num = size(wavefield_forward,4)
    receiver_num = size(receiver_position,1)
    u1 = SharedArray{Float64}(Nx, Ny, source_num)
    
    data_single = single_scattering(wavefield_forward, u0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    data_single = SharedArray(data_single)
    
    @inbounds @sync @distributed for ind = 1:source_num
        
        # adjoint of single scattering
        adj_source = data_single[end:-1:1,:,ind];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        utt = zeros(Nx, Ny, Nt)
        @views @. utt[:,:,2:end-1] = (wavefield_forward[:,:,3:end,ind]-2*wavefield_forward[:,:,2:end-1,ind]+wavefield_forward[:,:,1:end-2,ind]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3) * dt
        u1[:,:,ind] = grad0
        
    end
    
    u1 = Array(u1)
    u1 = sum(u1, dims=3)
    u1 = u1[:,:,1]
    u1 = reshape(u1, Nx*Ny,1)
    
    return u1
end

# This function is for compute the gradient of objective function (L2)

function compute_gradient(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        @views adj_source = data_forward - received_data[:,:,ind]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3) * dt
        grad[:,:,ind] = grad0
        
        # evaluate the objctive function
        obj_value[ind] = 0.5*norm(data_forward-received_data[:,:,ind],2).^2
    end
    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]

    obj_value = sum(obj_value)
    grad = reshape(grad, Nx*Ny,1)
    
    return grad, obj_value
end

function eval_obj_fn(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        # evaluate the objctive function
        obj_value[ind] = 0.5*norm(data_forward-received_data[:,:,ind],2).^2
    end

    obj_value = sum(obj_value)
    return obj_value
end

# This function is for compute the gradient of objective function (L2): Multi-channel
function compute_gradient_mc(received_data, u, Nc, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    u = reshape(u, Nx, Ny, Nc)
    c = u[:,:,1]
    rho = u[:,:,2]
    source_num = size(source_position,1)
    grad_c = SharedArray{Float64}(Nx, Ny, source_num)
    grad_rho = SharedArray{Float64}(Nx, Ny, source_num)
    grad = zeros(Nx, Ny, 2)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        @views adj_source = data_forward - received_data[:,:,ind]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = 0 .* u
        spatial_term = 0 .* u
        v = v[:,:,end:-1:1];
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = utt .* v;
        @views @. spatial_term[2:end-1,2:end-1,:] = (u[3:end,2:end-1,:]-u[2:end-1,2:end-1,:]).*(v[3:end,2:end-1,:]-v[2:end-1,2:end-1,:]) ./ h^2 + (u[2:end-1,3:end,:]-u[2:end-1,2:end-1,:]).*(v[2:end-1,3:end,:]-v[2:end-1,2:end-1,:]) ./ h^2;
        # grad_c[:,:,ind] = 2 .* sum(utt,dims=3) * dt ./(rho .* c.^3)
        # grad_rho[:,:,ind] = sum(utt,dims=3) * dt ./(rho .* c.^2) + sum(spatial_term,dims=3) * dt ./ (rho.^2);
        utt = sum(utt,dims=3) * dt
        spatial_term = sum(spatial_term,dims=3) * dt
        grad_c[:,:,ind] = 2 .* utt ./(rho .* c.^3)
        grad_rho[:,:,ind] = utt ./(rho .* c.^2) + spatial_term ./ (rho.^2);
        
        # evaluate the objctive function
        obj_value[ind] = 0.5*norm(data_forward-received_data[:,:,ind],2).^2
    end
    grad_c = Array(grad_c)
    grad_c = sum(grad_c, dims=3)
    grad_c = grad_c[:,:,1]

    grad_rho = Array(grad_rho)
    grad_rho = sum(grad_rho, dims=3)
    grad_rho = grad_rho[:,:,1]

    obj_value = sum(obj_value)
    grad[:,:,1] = grad_c
    grad[:,:,2] = grad_rho
    grad = grad[:]
    
    return grad, obj_value
end

function eval_obj_fn_mc(received_data, u, Nc, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    u = reshape(u, Nx, Ny, Nc)
    c = u[:,:,1]
    rho = u[:,:,2]
    source_num = size(source_position,1)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        # evaluate the objctive function
        obj_value[ind] = 0.5*norm(data_forward-received_data[:,:,ind],2).^2
    end

    obj_value = sum(obj_value)
    return obj_value
end