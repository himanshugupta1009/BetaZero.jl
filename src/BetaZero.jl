module BetaZero

using BSON
using DataStructures
using Distributed
using Flux
using MCTS
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Parameters
using POMDPs
using POMDPTools
using ProgressMeter
using Random
using Statistics
using StatsBase
using UnicodePlots

include("belief_mdp.jl")
include("representation.jl")

export
    BetaZeroSolver,
    BetaZeroPolicy,
    BetaZeroNetworkParameters,
    BeliefMDP


mutable struct BetaZeroPolicy <: POMDPs.Policy
    network::Chain
    planner::AbstractMCTSPlanner
end


@with_kw mutable struct BetaZeroNetworkParameters
    input_size = (30,30,5)
    training_epochs::Int = 1000 # Number of SGD training updates
    n_samples::Int = 10_000 # Number of samples (i.e., time steps) to use during training + validation
    normalize_target::Bool = true # Normalize target data to standard normal (0 mean)
    training_split::Float64 = 0.8 # Training / validation split (Default: 80/20)
    batchsize::Int = 512
    learning_rate::Float64 = 0.01 # Learning rate for ADAM optimizer during training
    λ_regularization::Float64 = 0.0001 # Parameter for L2-norm regularization
    loss_func::Function = Flux.Losses.mae # MAE works well for problems with large returns around zero, and spread out otherwise.
    device = gpu
    verbose_update_frequency::Int = training_epochs # Frequency of printed training output
    verbose_plot_frequency::Number = 10 # Frequency of plotted training/validation output
end


@with_kw mutable struct BetaZeroSolver <: POMDPs.Solver
    n_iterations::Int = 10 # BetaZero policy iterations (primary outer loop).
    n_data_gen::Int = 100 # Number of episodes to run for training/validation data generation.
    n_evaluate::Int = 0 # Number of episodes to run for network evaluation and comparison.
    n_holdout::Int = 50 # Number of episodes to run for a holdout test set (on a fixed, non-training or evaluation set).
    n_buffer::Int = 5*n_data_gen # Number of simulations to keep data for network training (NOTE: each simulation has multiple time steps of data, not counted in this number)
    data_buffer::CircularBuffer = CircularBuffer(n_buffer) # Simulation data buffer for training (NOTE: each simulation has multiple time steps of data)
    λ_ucb::Real = 0.0 # Upper confidence bound parameter: μ + λσ # TODO: Remove?
    updater::POMDPs.Updater
    network_params::BetaZeroNetworkParameters = BetaZeroNetworkParameters() # parameters for training CNN
    belief_reward::Function = (pomdp::POMDP, b, a, bp)->0.0
    # TODO: belief_representation::Function (see `representation.jl` TODO: should it be a parameter or overloaded function?)
    tree_in_info::Bool = false
    mcts_solver::AbstractMCTSSolver = DPWSolver(n_iterations=100,
                                                check_repeat_action=true,
                                                exploration_constant=1.0, # 1.0
                                                k_action=2.0, # 10
                                                alpha_action=0.25, # 0.5
                                                k_state=10.0, # 10
                                                alpha_state=0.1, # 0.5
                                                tree_in_info=tree_in_info,
                                                show_progress=false,
                                                estimate_value=(bmdp,b,d)->0.0) # `estimate_value` will be replaced with a neural network lookup
    bmdp::Union{BeliefMDP,Nothing} = nothing # Belief-MDP version of the POMDP
    collect_metrics::Bool = true # Indicate that performance metrics should be collected.
    performance_metrics::Array = [] # TODO: store_metrics for NON-HOLDOUT runs.
    holdout_metrics::Array = [] # Metrics computed from holdout test set.
    accuracy_func::Function = (pomdp,belief,state,action,returns)->nothing # (returns Bool): Function to indicate that the decision was "correct" (if applicable)
    verbose::Bool = true # Print out debugging/training/simulation information during solving
end


@with_kw mutable struct BetaZeroTrainingData
    b = nothing # current belief
    π = nothing # current policy estimate (using N(s,a))
    z = nothing # final discounted return of the episode
end


# Needs BetaZeroSolver defined.
include("metrics.jl")


"""
Run @time on expression based on `verbose` flag.
"""
macro conditional_time(verbose, expr)
    esc(quote
        if $verbose
            @time $expr
        else
            $expr
        end
    end)
end


"""
The main BetaZero policy iteration algorithm.
"""
function POMDPs.solve(solver::BetaZeroSolver, pomdp::POMDP)
    fill_bmdp!(pomdp, solver)
    f_prev = initialize_network(solver)

    @conditional_time solver.verbose for i in 1:solver.n_iterations
        solver.verbose && println(); println("—"^40); println(); @info "BetaZero iteration $i/$(solver.n_iterations)"

        # 0) Evaluate performance on a holdout test set (never used for training or network selection).
        # run_holdout_test!(pomdp, solver, f_prev; outer_iter=i) # TODO: DEBUGGING
        run_holdout_test!(pomdp, solver, f_prev)

        # 1) Generate data using the best BetaZero agent so far: {[belief, return], ...}
        generate_data!(pomdp, solver, f_prev; outer_iter=i)

        # 2) Optimize neural network parameters with recent simulated data (to estimate value given belief).
        f_curr = train_network(deepcopy(f_prev), solver; verbose=solver.verbose)

        # 3) Evaluate BetaZero agent (compare to previous agent based on mean returns).
        # f_prev = evaluate_agent(pomdp, solver, f_prev, f_curr; outer_iter=i) # TODO: DEBUGGING
        f_prev = evaluate_agent(pomdp, solver, f_prev, f_curr; outer_iter=typemax(Int32)+i)
    end

    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(b, f_prev)
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    policy = BetaZeroPolicy(f_prev, mcts_planner)

    return policy
end


"""
Conver the `POMDP` to a `BeliefMDP` and set the `pomdp.bmdp` field.
"""
function fill_bmdp!(pomdp::POMDP, solver::BetaZeroSolver)
    solver.bmdp = BeliefMDP(pomdp, solver.updater, solver.belief_reward)
    return solver.bmdp
end


"""
Initialize policy & value network with random weights.
"""
initialize_network(solver::BetaZeroSolver) = initialize_network(solver.network_params)
function initialize_network(nn_params::BetaZeroNetworkParameters) # LeNet5
    input_size = nn_params.input_size
    filter = (5,5)
    num_filters1 = 6
    num_filters2 = 16
    out_conv_size = prod([input_size[1] - 2*(filter[1]-1), input_size[2] - 2*(filter[2]-1), num_filters2])
    num_dense1 = 120
    num_dense2 = 84
    out_dim = 1

    return Chain(
        Conv(filter, input_size[end]=>num_filters1, relu),
        Conv(filter, num_filters1=>num_filters2, relu),
        Flux.flatten,
        Dense(out_conv_size, num_dense1, relu),
        Dense(num_dense1, num_dense2, relu),
        Dense(num_dense2, out_dim),
        # Note: A normalization layer will be added during training (with the old layer removed before the next training phase).
    )
end


"""
Train policy & value neural network `f` using the latest `data` generated from online tree search (MCTS).
"""
function train_network(f, solver::BetaZeroSolver; verbose::Bool=false)
    nn_params = solver.network_params
    data = sample_data(solver.data_buffer, nn_params.n_samples) # sample `n_samples` from last `n_buffer` simulations.
    x_data, y_data = data.X, data.Y

    # Normalize target values close to the range of [-1, 1]
    if nn_params.normalize_target
        mean_y = mean(y_data)
        std_y = std(y_data)
        y_data = (y_data .- mean_y) ./ std_y
    end

    n_data = length(y_data)
    n_train = Int(n_data ÷ (1/nn_params.training_split))

    verbose && @info "Data set size: $n_data"

    perm = randperm(n_data)
    perm_train = perm[1:n_train]
    perm_valid = perm[n_train+1:n_data]

    x_train = x_data[:,:,:,perm_train]
    y_train = y_data[:, perm_train]

    x_valid = x_data[:,:,:,perm_valid]
    y_valid = y_data[:, perm_valid]

    # Put model/data onto GPU device
    device = nn_params.device
    x_train = device(x_train)
    y_train = device(y_train)
    x_valid = device(x_valid)
    y_valid = device(y_valid)

    if n_train < nn_params.batchsize
        batchsize = n_train
        @warn "Number of observations less than batch-size, decreasing the batch-size to $batchsize"
    else
        batchsize = nn_params.batchsize
    end

    train_data = Flux.Data.DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true)

    # Remove un-normalization layer (if added from previous iteration)
    # We want to train for values close to [-1, 1]
    if isa(f.layers[end], Function)
        f = Chain(f.layers[1:end-1]...)
    end

    # Put network on GPU for training
    f = device(f)

    sqnorm(x) = sum(abs2, x)
    penalty() = nn_params.λ_regularization*sum(sqnorm, Flux.params(f))
    accuracy(x, y) = mean(sign.(f(x)) .== sign.(y))
    loss(x, y) = nn_params.loss_func(f(x), y) + penalty()

    # TODO: Include action/policy vector and change loss to include CE-loss

    opt = ADAM(nn_params.learning_rate)
    θ = Flux.params(f)

    training_epochs = nn_params.training_epochs
    losses_train = []
    losses_valid = []
    accs_train = []
    accs_valid = []
    verbose && @info "Beginning training $(size(x_train))"
    for e in 1:training_epochs
        for (x, y) in train_data
            _, back = Flux.pullback(() -> loss(x, y), θ)
            Flux.update!(opt, θ, back(1.0f0))
        end
        loss_train = loss(x_train, y_train)
        loss_valid = loss(x_valid, y_valid)
        acc_train = accuracy(x_train, y_train)
        acc_valid = accuracy(x_valid, y_valid)
        push!(losses_train, loss_train)
        push!(losses_valid, loss_valid)
        push!(accs_train, acc_train)
        push!(accs_valid, acc_valid)
        if verbose && e % nn_params.verbose_update_frequency == 0
            println("Epoch: ", e, " Loss Train: ", loss_train, " Loss Val: ", loss_valid, " | Acc. Train: ", acc_train, " Acc. Val: ", acc_valid)
        end
        if e % nn_params.verbose_plot_frequency == 0
            plot(xlims=(1, training_epochs), ylims=(0, nn_params.normalize_target ? 1 : 2000)) # TODO: Generalize
            plot!(1:e, losses_train, label="training")
            plot!(1:e, losses_valid, label="validation")
            display(plot!())
        end
    end

    if nn_params.verbose_plot_frequency != Inf
        learning_curve = plot!()
        display(learning_curve)

        value_model = (cpu(f(x_valid))' .* std_y) .+ mean_y
        value_data = (cpu(y_valid)' .* std_y) .+ mean_y
        value_distribution = Plots.histogram(value_model, alpha=0.5, label="model", c=3)
        Plots.histogram!(value_data, alpha=0.5, label="data", c=4)
        display(value_distribution)
    end

    # Place network on the CPU (better GPU memory conservation when doing parallelized inference)
    f = cpu(f)

    # Clean GPU memory explicitly
    if device == gpu
        x_train = y_train = x_valid = y_valid = nothing
        GC.gc()
        Flux.CUDA.reclaim()
    end

    # Add un-normalization layer
    unnormalize = y -> (y .* std_y) .+ mean_y
    f = Chain(f.layers..., unnormalize)

    return f
end


"""
Evaluate the neural network `f` using the `belief` as input.
Note, inference is done on the CPU given a single input.
"""
function value_lookup(belief, f)
    b = input_representation(belief)
    b = Float32.(b)
    x = Flux.unsqueeze(b; dims=ndims(b)+1)
    y = f(x) # evaluate network `f`
    value = cpu(y)[1] # returns 1 element 1D array
    return value
end


"""
Compare previous and current neural networks using MCTS simulations.
Use upper confidence bound on the discounted return as the comparison metric.
"""
function evaluate_agent(pomdp::POMDP, solver::BetaZeroSolver, f_prev, f_curr; outer_iter=0)
    # Run a number of simulations to evaluate the two neural networks using MCTS (`f_prev` and `f_curr`)
    if solver.n_evaluate == 0
        solver.verbose && @info "Skipping network evaluations, selected newest network."
        return f_curr
    else
        solver.verbose && @info "Evaluting networks..."
        returns_prev = generate_data!(pomdp, solver, f_prev; inner_iter=solver.n_evaluate, outer_iter=outer_iter, store_data=false)[:G]
        returns_curr = generate_data!(pomdp, solver, f_curr; inner_iter=solver.n_evaluate, outer_iter=outer_iter, store_data=false)[:G]

        λ = solver.λ_ucb
        μ_prev, σ_prev = mean_and_std(returns_prev)
        μ_curr, σ_curr = mean_and_std(returns_curr)
        ucb_prev = μ_prev + λ*σ_prev
        ucb_curr = μ_curr + λ*σ_curr

        solver.verbose && @show ucb_curr, ucb_prev

        if ucb_curr > ucb_prev
            solver.verbose && @info "<<<< New network performed better >>>>"
            return f_curr
        else
            if solver.verbose && ucb_curr == ucb_prev
                @info "[IDENTICAL UCBs]"
            end
            solver.verbose && @info "---- Previous network performed better ----"
            return f_prev
        end
    end
end


"""
Generate training data using online MCTS with the best network so far `f` (parallelized across episodes).
"""
function generate_data!(pomdp::POMDP, solver::BetaZeroSolver, f; outer_iter::Int=0, inner_iter::Int=solver.n_data_gen, store_metrics::Bool=false, store_data::Bool=true)
    # Confirm that network is on the CPU for inference
    f = cpu(f)

    # Run MCTS to generate data using the neural network `f`
    isnothing(solver.bmdp) && fill_bmdp!(pomdp, solver)
    solver.mcts_solver.estimate_value = (bmdp,b,d)->value_lookup(b, f)
    mcts_planner = solve(solver.mcts_solver, solver.bmdp)
    up = solver.updater
    ds0 = POMDPs.initialstate_distribution(pomdp)
    collect_metrics = solver.collect_metrics
    accuracy_func = solver.accuracy_func
    tree_in_info = solver.tree_in_info

    # (nprocs() < nbatches) && addprocs(nbatches - nprocs())
    solver.verbose && @info "Number of processes: $(nprocs())"

    progress = Progress(inner_iter)
    channel = RemoteChannel(()->Channel{Bool}(), 1)

    @async while take!(channel)
        next!(progress)
    end

    @time parallel_data = pmap(i->begin
            seed = parse(Int, string(outer_iter, lpad(i, length(digits(inner_iter)), '0'))) # 1001, 1002, etc. for BetaZero outer_iter=1
            Random.seed!(seed)
            # @info "Generating data ($i/$(inner_iter)) with seed ($seed)"
            s0 = rand(ds0)
            b0 = POMDPs.initialize_belief(up, ds0)
            data, metrics = run_simulation(pomdp, mcts_planner, up, b0, s0; collect_metrics, accuracy_func, tree_in_info)
            B = []
            Z = []
            # Π = []
            discounted_return = data[1].z
            for d in data
                push!(B, d.b)
                push!(Z, d.z)
                # push!(Π, d.π) # TODO.
            end
            put!(channel, true) # trigger progress bar update
            B, Z, metrics, discounted_return
        end, 1:inner_iter)

    put!(channel, false) # tell printing task to finish

    beliefs = vcat([d[1] for d in parallel_data]...) # combine all beliefs
    returns = vcat([d[2] for d in parallel_data]...) # combine all returns
    metrics = vcat([d[3] for d in parallel_data]...) # combine all metrics
    G = vcat([d[4] for d in parallel_data]...) # combine all final returns

    if store_metrics
        push!(solver.performance_metrics, metrics...)
    end

    # Much faster than `cat(belief...; dims=4)`
    belief = beliefs[1]
    X = Array{Float32}(undef, size(belief)..., length(beliefs))
    for i in eachindex(beliefs)
        # Generalize for any size matrix (equivalent to X[:,:,:,i] = beliefs[i] for 3D matrix)
        setindex!(X, beliefs[i], map(d->1:d, size(belief))..., i)
    end
    Y = reshape(Float32.(returns), 1, length(returns))

    data = (X=X, Y=Y, G=G)

    if store_data
        # Store data in buffer for training
        push!(solver.data_buffer, data)
    end

    return data
end


"""
Uniformly sample data from buffer (with replacement).
Note that the buffer is per-simulation with each simulation having multiple time steps.
We want to sample `n` individual time steps across the simulations.
"""
function sample_data(data_buffer::CircularBuffer, n::Int)
    sim_times = map(d->length(d.Y), data_buffer) # number of time steps in each simulation
    data_buffer_indices = 1:length(data_buffer)
    sampled_sims_indices = sample(data_buffer_indices, Weights(sim_times), n; replace=true) # weighted based on num. steps per sim (to keep with __overall__ uniform across time steps)
    belief_size = size(data_buffer[1].X)[1:end-1]
    X = Array{Float32}(undef, belief_size..., n)
    Y = Array{Float32}(undef, 1, n)
    G = Vector{Float32}(undef, n)
    for (i,sim_i) in enumerate(sampled_sims_indices)
        sim = data_buffer[sim_i]
        T = length(sim.Y)
        t = rand(1:T) # uniformly sample time from this simulation
        belief_size_span = map(d->1:d, belief_size) # e.g., (1:30, 1:30, 1:5)
        setindex!(X, getindex(sim.X, belief_size_span..., t), belief_size_span..., i) # general for any size matrix e.g., X[:,;,:,i] = sim.X[:,:,:,t]
        Y[i] = sim.Y[t]
        G[i] = sim.G[sim_i]
    end
    return (X=X, Y=Y, G=G)
end


"""
Compute the discounted `γ` returns from reward vector `R`.
"""
function compute_returns(R; γ=1)
    T = length(R)
    G = zeros(T)
    for t in reverse(1:T)
        G[t] = t==T ? R[t] : G[t] = R[t] + γ*G[t+1]
    end
    return G
end


"""
Run single simulation using a belief-MCTS policy on the original POMDP (i.e., notabily, not on the belief-MDP).
"""
function run_simulation(pomdp::POMDP, policy::POMDPs.Policy, up::POMDPs.Updater, b0, s0;
                        max_steps=100, collect_metrics::Bool=false, accuracy_func::Function=(args...)->nothing, tree_in_info::Bool=false)
    rewards::Vector{Float64} = [0.0]
    data = [BetaZeroTrainingData(b=input_representation(b0))]
    local action
    local T
    trees::Vector{Union{MCTS.DPWTree,MCTS.MCTSTree}} = [] # TODO: Other MCTS Tree types
    for (a,r,bp,t,info) in stepthrough(pomdp, policy, up, b0, s0, "a,r,bp,t,action_info", max_steps=max_steps)
        # @info "Simulation time step $t"
        T = t
        action = a
        push!(rewards, r)
        push!(data, BetaZeroTrainingData(b=input_representation(bp)))
        if tree_in_info
            push!(trees, info[:tree])
        end
    end

    γ = POMDPs.discount(pomdp)
    G = compute_returns(rewards; γ=γ)

    for (t,d) in enumerate(data)
        d.z = G[t]
    end

    metrics = collect_metrics ? compute_performance_metrics(pomdp, data, accuracy_func, b0, s0, action, trees, T) : nothing

    return data, metrics
end


"""
Method to collect performance and validation metrics during BetaZero policy iteration.
Note, user defines `solver.accuracy_func` to determine the accuracy of the final decision (if applicable).
"""
function compute_performance_metrics(pomdp::POMDP, data, accuracy_func::Function, b0, s0, action, trees, T)
    # - mean discounted return over time
    # - accuracy over time (i.e., did it make the correct decision, if there's some notion of correct)
    # - number of actions (e.g., number of drills for mineral exploration)
    returns = [d.z for d in data]
    discounted_return = returns[1]
    accuracy = accuracy_func(pomdp, b0, s0, action, returns) # NOTE: Problem specific, provide function to compute this
    return (discounted_return=discounted_return, accuracy=accuracy, num_actions=T, trees=trees, action=action)
end


"""
Run a test on a holdout set to collect performance metrics during BetaZero policy iteration.
"""
function run_holdout_test!(pomdp::POMDP, solver::BetaZeroSolver, f; outer_iter::Int=0)
    if solver.n_holdout > 0
        solver.verbose && @info "Running holdout test..."
        returns = generate_data!(pomdp, solver, f; inner_iter=solver.n_holdout, outer_iter=outer_iter, store_metrics=true, store_data=false)[:G]
        solver.verbose && display(UnicodePlots.histogram(returns))
        μ, σ = mean_and_std(returns)
        push!(solver.holdout_metrics, (mean=μ, std=σ, returns=returns))
        solver.verbose && @show μ, σ
    end
end


"""
Save performance metrics to a file.
"""
function save_metrics(solver::BetaZeroSolver, filename::String)
    metrics = solver.performance_metrics
    BSON.@save filename metrics
end


"""
Save policy to file (MCTS planner and network objects together).
"""
function save_policy(policy::BetaZeroPolicy, filename::String)
    BSON.@save "$filename" policy
end


"""
Load policy from file (MCTS planner and network objects together).
"""
function load_policy(filename::String)
    BSON.@load "$filename" policy
    return policy
end


"""
Save just the neural network to a file.
"""
function save_network(policy::BetaZeroPolicy, filename::String)
    network = policy.network
    BSON.@save "$filename" network
end


"""
Load just the neural network from a file.
"""
function load_network(filename::String)
    BSON.@load "$filename" network
    return network
end


"""
Get action from BetaZero policy (online MCTS using value & policy network).
"""
POMDPs.action(policy::BetaZeroPolicy, b) = action(policy.planner, b)
POMDPTools.action_info(policy::BetaZeroPolicy, b; tree_in_info=false) = POMDPTools.action_info(policy.planner, b; tree_in_info)


end # module
