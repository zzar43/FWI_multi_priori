using Printf

# backtrack linesearch
function linesearch_back(u_bar, u0, f_u0, eval_fn, eta; alpha=1, maxSearch=5)
    u1 = 0 * u0
    for i = 1:maxSearch
        @printf "    Line search time: %1d\n" i
        u1 = u0 + alpha * (u_bar - u0)
        f_u1 = eval_fn(u1)
        @printf "    alpha = %1.3e,  f_u0 = %1.5e,  f_u1 = %1.5e\n" alpha f_u0 f_u1
        if f_u1 < f_u0 && (i < maxSearch)
            println("    Line search succeed.")
            break;
        elseif f_u1 >= f_u0 && (i == maxSearch)
            u1 = copy(u0)
            println("    Line search failed.")
        else
            alpha = eta * alpha
        end
    end
    return u1
end

# l-BFGS matrix computation
function compute_H_ku(u, S, Y, R, D)
    u = u[:]
    s0 = S[:,end]
    y0 = Y[:,end]
    R_inv = inv(R)
    gamma = (y0'*s0)[1] / (y0'*y0)[1]
    A = [S'; gamma * Y'] * u
    A = [R_inv' * (D + gamma * Y' * Y) * R_inv -R_inv'; -R_inv 0*R] * A
    A = [S gamma * Y] * A
    A = gamma .* u + A
    return A
end

function compute_B_ku(u, S, Y, L, D)
    u = u[:]
    s0 = S[:,end]
    y0 = Y[:,end]
    sigma = (y0'*s0)[1] / (s0'*s0)[1]
    A = [sigma * S'; Y'] * u
    B = [sigma * S' * S L; L' -D]
    A = inv(B) * A
    A = [sigma*S Y] * A
    A = sigma .* u - A
    return A
end