module IsingModelTest

    export IsingModel, Ensemble, Plot2D, Gif2D, GetUpSpins, CentralMoments, EnsProbUpSpins

    using Colors, Plots, StatsBase, Parameters, Distributions, StatsBase, LinearAlgebra;

    """
    Define the Ising model struct.
    """
    @with_kw struct IsingModel
        N::Int64 = 100
        h::Float64
        J::Float64
        p0::Float64 = 0.5 # initial prob of spin up
        periodic::Int64 = 1
        mf::Int64 = 0
        fully_connected::Int64 = 0
    end

    """
    Define the struct for an ensemble of sims.
    """
    @with_kw struct Ensemble
        IM::IsingModel
        e_size::Int64 = 1 # size of ensemble
        method::String = "Glauber" # or "SSA"
        tₘ::Real = 10^5 # no units of time
        Δt::Real = 10^2 # no units of time
        ts::Real = 1.0 # timescale: real time τ = ts*t
        sims = SimRun(method,e_size,tₘ,Δt,ts,IM)
    end

    """
    Choice of algorithm
    """
    function SimRun(alg::String,e_size::Int64,tₘ::Real,Δt::Real,ts::Real,IM::IsingModel)
        if alg == "Glauber"
            return Glauber(e_size,tₘ,Δt,ts,IM)
        elseif alg == "SSA"
            return SSA(e_size,tₘ,Δt,ts,IM)
        else
            error(alg, " is not an appropriate choice of algorithm.
            Please choose either 'Glauber' or 'SSA'.")
        end
    end

    """
    Make the initial NxN matrix of spins.
    """
    function S_make(p0::Float64, N::Int64)
        if p0 > 1 || p0 < 0
            error("p must be bounded between 0 and 1")
        else
            return ones(N,N)-2*sample([1,0],Weights([1-p0,p0]),(N,N))
        end
    end

    """
    Define the effective field at each spin.
    """
    function hₑ(i::Int64, j::Int64, S::Matrix{Float64}, IM::IsingModel, periodic::Int64)
        @unpack J, h, N, fully_connected = IM;
        if fully_connected == 1
            return 2*h + (J/N)*(sum(S)-S[i,j]) # J scaled by N.
        elseif fully_connected == 0 && periodic == 0
            if i==1 && j==1
                return 2*h + J*(S[i+1,j]+S[i,j+1])
            elseif i==N && j==N
                return 2*h + J*(S[i-1,j]+S[i,j-1])
            elseif i==1 && j==N
                return 2*h + J*(S[i+1,j]+S[i,j-1])
            elseif i==N && j==1
                return 2*h + J*(S[i-1,j]+S[i,j+1])
            elseif i==1
                return 2*h + J*(S[i+1,j]+S[i,j+1]+S[i,j-1])
            elseif j==1
                return 2*h + J*(S[i+1,j]+S[i-1,j]+S[i,j+1])
            elseif i==N
                return 2*h + J*(S[i-1,j]+S[i,j+1]+S[i,j-1])
            elseif j==N
                return 2*h + J*(S[i+1,j]+S[i-1,j]+S[i,j-1])
            else
                return 2*h + J*(S[i+1,j]+S[i-1,j]+S[i,j+1]+S[i,j-1])
            end
        else # is periodic BC
            return 2*h + J*(S[mod1(i+1,N),j]+S[mod1(i-1,N),j]+S[i,mod1(j+1,N)]+S[i,mod1(j-1,N)])
        end
    end

    """
    Effective field in MF setting.
    """
    function hₘ(i::Int64, j::Int64, S::Matrix{Float64}, IM::IsingModel, periodic::Int64)
        @unpack J, h, N, fully_connected = IM;
        m = mean(S);
        Jm = J*m;
        if fully_connected == 1
            return 2*h + J*m # J scaled by N.
        elseif fully_connected == 0 && periodic == 0
            if i==1 && j==1
                return 2*h + 2*Jm
            elseif i==N && j==N
                return 2*h + 2*Jm
            elseif i==1 && j==N
                return 2*h + 2*Jm
            elseif i==N && j==1
                return 2*h + 2*Jm
            elseif i==1
                return 2*h + 3*Jm
            elseif j==1
                return 2*h + 3*Jm
            elseif i==N
                return 2*h + 3*Jm
            elseif j==N
                return 2*h + 3*Jm
            else
                return 2*h + 4*Jm
            end
        else # is periodic BC
            return 2*h + 4*Jm
        end
    end

    """
    Energy difference calculation
    """
    function ΔH(i::Int64, j::Int64, S::Matrix{Float64}, IM::IsingModel, periodic::Int64=1, mf::Int64=0)
        if mf == 0
            Δe = hₑ(i, j, S, IM, periodic)*S[i,j]
        elseif mf == 1
            Δe = hₘ(i, j, S, IM, periodic)*S[i,j]
        else
            error("mf must be 1 or 0.")
        end
        return Δe
    end

    """
    Logistic rule.
    """
    function LogisticRule(E₀::Float64)
        return (1+exp(E₀))^-1
    end

    """
    Glauber simulation.
    """
    function Glauber(ens_its::Int64,tₘ::Int64,Δt::Int64,μ₁::Float64,IM::IsingModel)
        @unpack p0, N, mf, periodic = IM;
        ens_store = Array{Vector{Matrix{Float64}}}(undef, ens_its);
        for it in 1:ens_its
            S = S_make(p0, N); # make initial config
            S_temp = copy(S); # use S_temp for updating.
            store = Array{Matrix{Float64}}(undef, tₘ÷Δt + 1);
            store[1] = copy(S_temp);
            for n in 1:tₘ+1
                i,j = Int.(map(round,rand(2).*(N-1))) .+ [1,1]; # agent chosen at random
                E₀ = ΔH(i,j,S_temp,IM,periodic,mf) # periodic BC
                pr = LogisticRule(E₀);
                weights = [pr,1-pr]; # flip, don't flip
                flip = sample([1,0],Weights(weights));
                if flip == 1 # if chosen to flip perform
                    S_temp[i,j] = -S_temp[i,j]
                end
                if n%Δt == 0 # store every Δt steps
                    store[(n÷Δt)] = copy(S_temp)
                end
            end
            ens_store[it] = store
        end
        return ens_store
    end

    """
    Propensity for SSA
    """
    function propensity(S::Matrix{Float64},IM::IsingModel,γ::Float64)
        @unpack N, mf, periodic = IM;
        γ = γ/(N^2); # normalise the timescale to match Glauber.
        props = zeros(size(S))
        for i in 1:size(S)[1]
            for j in 1:size(S)[2]
                E₀ = ΔH(i,j,S,IM,periodic,mf); # periodic BC
                pr = LogisticRule(E₀);
                props[i,j] = γ*pr;
            end
        end
        return props
    end

    """
    SSA simulations
    """
    function SSA(ens_its::Int64,τₘ::Real,dτ::Real,γ::Real,IM::IsingModel)
        @unpack p0, N, mf, periodic = IM;
        ens_store = Array{Vector{Matrix{Float64}}}(undef, ens_its);
        for it in 1:ens_its
            S = S_make(p0, N); # make initial config
            S_temp = copy(S); # use S_temp for updating.
            store = Array{Matrix{Float64}}(undef, Int(τₘ÷dτ)+1);
            τ = 0.0;
            sim = 0;
            while τ < τₘ
                props = propensity(S_temp,IM,γ);
                props_flat = vec(transpose(props)); # concatted rows
                w0 = sum(props);
                r1, r2 = rand(2);
                u = (1/w0)*log(1/r1);
                k = findfirst(x -> x>=r2*w0,cumsum(props_flat));
                n = 0;
                while τ+n*dτ < τ+u
                    if τ+n*dτ > τₘ
                        store[Int(τₘ÷dτ)+1] = copy(S_temp)
                        break
                    else
                        store[Int((τ+n*dτ)÷dτ)+1] = copy(S_temp)
                        n+=1;
                    end
                end
                new_i = ((k-1)÷N)+1;
                new_j = k-(new_i-1)*N;
                S_temp[new_i,new_j] = -S_temp[new_i,new_j];
                τ += u;
                if τ > τₘ
                    store[Int(τₘ÷dτ)+1] = copy(S_temp)
                end
                sim += 1;
                if sim%1000==0
                    println(τ)
                end
            end
            ens_store[it] = store;
        end
        return ens_store
    end

    """
    Plot slice at some time.
    """
    function Plot2D(t::Int64, m::Int64, ens::Ensemble) # m = sim set
        @unpack method, Δt, tₘ, ts, sims = ens;
        if method == "Glauber" # then t is timeslice
            plot(Gray.(sims[m][t]))
        elseif method == "SSA" # then t is the actual time
            n = Int((t/ts)÷Δt)
            if n <= tₘ
                plot(Gray.(sims[m][n]))
            else
                error("Plot time must be within the simulated times.")
            end
        else
            error("Simulation method chosen is invalid.")
        end
    end

    """
    IM gif maker.
    """
    function Gif2D(m::Int64, ens::Ensemble, fname::String, frames::Real=40)
        @unpack sims = ens;
        anim = @animate for i ∈ 1:length(sims[m])
            plot(Gray.(sims[m][i]))
        end
        return gif(anim, join([fname,".gif"]), fps = frames)
    end

    """
    Take the trajectories of up spins from each set of sims in an ensemble.
    """
    function GetUpSpins(ens::Ensemble, norm::Int64=0)
        @unpack IM, sims = ens
        @unpack N = IM
        store = Array{Float64,2}(undef, length(sims), length(sims[1]));#Array{Array{Float64,1}}
        for m in 1:length(sims)
            for t in 1:length(sims[1])
                store[m,t] = count(j->(j==1.0),sims[m][t])
            end
        end
        if norm == 0
            return store
        elseif norm == 1
            return store/N^2
        else
            error("norm must take a value of 0 or 1.")
        end
    end

    """
    Get the central moments from the ensemble
    """
    function CentralMoments(ens::Ensemble,n::Int64,norm::Int64=0)
        upspins = GetUpSpins(ens, norm)
        store = Array{Float64,1}(undef, length(upspins[1,:]))
        if n == 1
            for t in 1:length(upspins[1,:])
                store[t] = mean(upspins[:,t])
            end
        else
            for t in 1:length(upspins[1,:])
                store[t] = moment(upspins[:,t],n)
            end
        end
        # need to sort out times
        @unpack Δt, ts = ens;
        no_unit_t = LinRange(0:Δt:length(upspins[1,:])-1);
        times = no_unit_t * ts;
        return (times, store)
    end

    """
    Get prob distribution at all times for an ensemble
    """
    function EnsProbUpSpins(ens::Ensemble, T::Int64)
        UpSpins = GetUpSpins(ens)
        mod_bins = LinRange(minimum(UpSpins[:,T])-0.5:1:maximum(UpSpins[:,T])+0.5);
        mid_pts = LinRange(minimum(UpSpins[:,T]):1:maximum(UpSpins[:,T]));
        bin_vals = normalize(fit(Histogram, UpSpins[:,T], mod_bins), mode=:probability).weights;
        return (mid_pts, bin_vals)
    end

end # module end
