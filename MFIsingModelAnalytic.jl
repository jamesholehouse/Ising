module MFIsingAnalytic

    export AIM, prob, SSprob

    using Plots, Parameters, LinearAlgebra, Distributions, StatsBase

    """
    Define the Analytic Ising Model struct
    """
    @with_kw struct AIM
        N::Int64 = 100 # number of agents
        γ::Float64 = 1.0 # timescale
        h::Float64
        J::Float64
        z::Float64 = 1.0 # could generalise to be 2*d for NN.
        pars::Vector{Float64} = [γ, h, J, z]
        A::Matrix{Float64} = make_TRM(pars, N)
        λ::Array{Complex{BigFloat}} = GetEigVals(A)
        q_arr::Vector{Vector{Complex{BigFloat}}} = [GetOrthoQ(pars,N,λ[j]) for j in 1:N+1]
        p_arr::Vector{Vector{Complex{BigFloat}}} = [GetOrthoP(pars,N,λ[j]) for j in 1:N+1]
        den_prod::Vector{Complex{BigFloat}} = [prod([λ[i]-λ[j] for j in filter!(e->e≠i,[j for j in 1:N+1])]) for i in 1:N+1]
        As::Vector{BigFloat} = [a(pars,N,j) for j in 1:N]
        Bs::Vector{BigFloat} = [b(pars,N,j) for j in 0:N-1]
    end

    """
    Define rate function for N₊ prod
    """
    function r1(pars::Vector{Float64}, N::Int64, n::Int64)
        γ, h, J, z = pars
        return convert(BigFloat,γ*(N-n)*(1+exp(-2*(h+J*z*((2*n-N)/N))))^-1)::BigFloat
    end
    # rescaled definition in solution
    a(pars,N,n) = r1(pars,N,n-1);

    """
    Define rate function for N₋ prod
    """
    function r2(pars::Vector{Float64}, N::Int64, n::Int64)
        γ, h, J, z = pars
        return convert(BigFloat,γ*n*(1+exp(2*(h+J*z*((2*n-N)/N))))^-1)::BigFloat
    end
    # rescaled definition in solution
    b(pars,N,n) = r2(pars,N,n+1);

    """
    Make the transition rate matrix
    """
    function make_TRM(pars::Vector{Float64}, N::Int64)
        A = zeros(N+1, N+1);
        for i in 1:size(A)[1]
            for j in 1:size(A)[2]
                if i == 1 && j == 1
                    A[1,1] = - a(pars,N,1)
                elseif i == j && i>1
                    A[i,i] = -(a(pars,N,i)+b(pars,N,i-2))
                elseif i == j+1
                    A[i,j] = a(pars,N,j)
                elseif i == j-1
                    A[i,j] = b(pars,N,i-1)
                else
                    continue
                end
            end
        end
        return A::Matrix{Float64}
    end

    """
    Get eigenvalues
    """
    function GetEigVals(A::Matrix{Float64})
        λ = convert(Array{Complex{BigFloat}}, reverse(eigvals(A)));
        λ[1] = 0.0 # explicitly code in the zero for the s.s.
        return λ::Array{Complex{BigFloat}}
    end

    """
    Get the p orthogonal polynomials
    """
    function GetOrthoP(pars::Vector{Float64}, N::Int64, λᵢ::Complex{BigFloat})
        p = Array{Complex{BigFloat},1}(undef, N+1);
        p[1] = 1.0; p[2] = λᵢ+a(pars,N,1);
        for i in 3:N+1
            p[i] = (λᵢ+a(pars,N,i-1)+b(pars,N,i-3))*p[i-1] - b(pars,N,i-3)*a(pars,N,i-2)*p[i-2]
        end
        return p::Vector{Complex{BigFloat}}
    end

    """
    Get the q orthogonal polynomials
    """
    function GetOrthoQ(pars::Vector{Float64}, N::Int64, λᵢ::Complex{BigFloat})
        q = Array{Complex{BigFloat},1}(undef, N+3);
        q[N+3] = 1.0; q[N+2] = λᵢ + b(pars,N,N-1);
        for i in reverse([j for j in 3:N+1])
            q[i] = (λᵢ + a(pars,N,i-1)+b(pars,N,i-3))*q[i+1] - b(pars,N,i-2)*a(pars,N,i-1)*q[i+2]
        end
        return q::Vector{Complex{BigFloat}}
    end

    """
    Define the sum of the elements from the solution
    """
    function sum_elems(λᵢ::Complex{BigFloat}, t::Float64, m::Int64, m₀::Int64, pars::Array{Float64,1}, N::Int64, p_arrᵢ::Vector{Complex{BigFloat}}, q_arrᵢ::Vector{Complex{BigFloat}}, den_prodᵢ::Complex{BigFloat})
        return exp(λᵢ*t)*p_arrᵢ[m+1]*q_arrᵢ[m₀+3]/den_prodᵢ::Complex{BigFloat}
    end

    """
    Define P(m,t|m₀)
    """
    function pm(t::Float64, IM::AIM, m::Int64, m₀::Int64)
        @unpack λ, pars, N, As, Bs, q_arr, p_arr, den_prod = IM
        if m<m₀
            return prod(Bs[m+1:m₀])*sum([sum_elems(λ[i], t, m, m₀, pars, N, p_arr[i], q_arr[i], den_prod[i]) for i in 1:N+1])::Complex{BigFloat}
        elseif m==m₀
            return sum([sum_elems(λ[i], t, m, m, pars, N, p_arr[i], q_arr[i], den_prod[i]) for i in 1:N+1])::Complex{BigFloat}
        else
            return prod(As[m₀+1:m])*sum([sum_elems(λ[i], t, m₀, m, pars, N, p_arr[i], q_arr[i], den_prod[i]) for i in 1:N+1])::Complex{BigFloat}
        end
    end

    """
    Define the probability distribution return function from a initial distribution.
    """
    function prob(IM::AIM, t::Float64, q_init_D::Distribution{Univariate, Discrete})
        @unpack N = IM
        q_init = pdf(q_init_D)
        pmt = Array{Complex{BigFloat}}(undef,N+1)
        for i in 1:N+1 # loop over the m0's
            pmt[i] = sum([q_init[n+1]*pm(t, IM, i-1, n) for n in 0:N])
        end
        return (LinRange(0.0,N,N+1),real(pmt))::Tuple{LinRange{Float64}, Vector{BigFloat}}
    end

    """
    Define the probability distribution return function from a precise value of m₀.
    """
    function prob(IM::AIM, t::Float64, m₀::Int64)
        @unpack N = IM
        return (LinRange(0.0,N,N+1),real([pm(t, IM, m, m₀) for m in 0:N]))::Tuple{LinRange{Float64}, Vector{BigFloat}}
    end

    """
    Define the probability distribution return function for N/2 value of m₀.
    """
    function prob(IM::AIM, t::Float64)
        @unpack N = IM
        m₀ = round(Int64,N/2)
        return (LinRange(0.0,N,N+1),real([pm(t, IM, m, m₀) for m in 0:N]))::Tuple{LinRange{Float64}, Vector{BigFloat}}
    end

    """
    Return the steady state distribution
    """
    function SSprob(IM::AIM)
        @unpack As, Bs, N = IM
        ps = Vector{BigFloat}(undef, N+1)
        for m in 2:N # use the product rule
            ps[m] = prod(As[1:m-1])*prod(Bs[m:N])
        end
        ps[1] = prod(Bs[1:N]) # do product for the B's
        ps[N+1] = prod(As[1:N]) # do product for the A's
        return (LinRange(0.0,N,N+1),ps/sum(ps))::Tuple{LinRange{Float64}, Vector{BigFloat}}
    end

end # module end
