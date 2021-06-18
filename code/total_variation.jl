# evaluate tv(u)
function eval_tv(u, Nx, Ny)
    u = reshape(u, Nx, Ny)
    tv_u = 0
    for i = 1:Nx-1
        for j = 1:Ny-1
            tv_u += sqrt((u[i+1,j]-u[i,j])^2 + (u[i,j+1]-u[i,j])^2)
        end
    end
    for j = 1:Ny-1
        tv_u += abs(u[Nx,j+1]-u[Nx,j])
    end
    for i = 1:Nx-1
        tv_u += abs(u[i+1,Ny]-u[i,Ny])
    end
    return tv_u
end

# evaluate vtv(u)
function eval_vtv(u, Nx, Ny, Nc)
    u = reshape(u, Nx, Ny, Nc)
    tv_u = 0
    for i = 1:Nx-1
        for j = 1:Ny-1
            for k = 1:Nc
                tv_u += sqrt((u[i+1,j,k]-u[i,j,k])^2 + (u[i,j+1,k]-u[i,j,k])^2)
            end
        end
    end
    for j = 1:Ny-1
        for k = 1:Nc
            tv_u += sqrt((u[Nx,j+1,k]-u[Nx,j,k])^2)
        end
    end
    for i = 1:Nx-1
        for k = 1:Nc
            tv_u += sqrt((u[i+1,Ny,k]-u[i,Ny,k])^2)
        end
    end
    return tv_u
end


# subgradient of tv(u)
function subgrad_tv(u, Nx, Ny)
    u = reshape(u, Nx, Ny)
    subgrad_tv_u = zeros(Nx, Ny)
    for i = 1:Nx-1
        for j = 1:Ny-1
            fij = sqrt((u[i+1,j]-u[i,j])^2 + (u[i,j+1]-u[i,j])^2)
            if fij != 0
                subgrad_tv_u[i+1,j] += (u[i+1,j]-u[i,j]) / fij
                subgrad_tv_u[i,j+1] += (u[i,j+1]-u[i,j]) / fij
                subgrad_tv_u[i,j] += -(u[i+1,j]-2*u[i,j]+u[i,j+1]) /fij
            end
        end
    end
    for j = 1:Ny-1
        fij = u[Nx,j+1]-u[Nx,j]
        if fij != 0
            subgrad_tv_u[Nx,j+1] += (fij)/abs(fij)*1
            subgrad_tv_u[Nx,j] += (fij)/abs(fij)*(-1)
        end
    end
    for i = 1:Nx-1
        fij = u[i+1,Ny]-u[i,Ny]
        if fij != 0
            subgrad_tv_u[i+1,Ny] += (fij)/abs(fij)*1
            subgrad_tv_u[i,Ny] += (fij)/abs(fij)*(-1)
        end
    end
    return subgrad_tv_u
end

# subgradient of vtv(u)
function subgrad_vtv(u, Nx, Ny, Nc)
    u = reshape(u, Nx, Ny, Nc)
    subgrad_tv_u = zeros(Nx, Ny, Nc)
    for i = 1:Nx-1
        for j = 1:Ny-1
            fij = 0
            for k = 1:Nc
                fij += sqrt((u[i+1,j,k]-u[i,j,k])^2 + (u[i,j+1,k]-u[i,j,k])^2)
            end
            if fij != 0
                for k = 1:Nc
                    subgrad_tv_u[i+1,j,k] += (u[i+1,j,k]-u[i,j,k]) / fij
                    subgrad_tv_u[i,j+1,k] += (u[i,j+1,k]-u[i,j,k]) / fij
                    subgrad_tv_u[i,j,k] += -(u[i+1,j,k]-2*u[i,j,k]+u[i,j+1,k]) /fij
                end
            end
        end
    end
    for j = 1:Ny-1
        fij = 0
        for k = 1:Nc
            fij += sqrt((u[Nx,j+1,k]-u[Nx,j,k])^2)
        end
        if fij != 0
            for k = 1:Nc
                subgrad_tv_u[Nx,j+1,k] += (u[Nx,j+1,k]-u[Nx,j,k]) / fij
                subgrad_tv_u[Nx,j,k] += -(u[Nx,j+1,k]-u[Nx,j,k]) / fij
            end
        end
    end
    for i = 1:Nx-1
        fij = 0
        for k = 1:Nc
            fij += sqrt((u[i+1,Ny,k]-u[i,Ny,k])^2)
        end
        if fij != 0
            for k = 1:Nc
                subgrad_tv_u[i+1,Ny,k] += (u[i+1,Ny,k]-u[i,Ny,k]) / fij
                subgrad_tv_u[i,Ny,k] += -(u[i+1,Ny,k]-u[i,Ny,k]) / fij
            end
        end
    end
    subgrad_tv_u = subgrad_tv_u[:]
    return subgrad_tv_u
end