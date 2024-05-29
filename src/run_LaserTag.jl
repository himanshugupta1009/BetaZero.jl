using Distributed
parallel_flag = false

if !parallel_flag
    using BetaZero
    using DMUStudent.HW6
    
    pomdp = LaserTagPOMDP()
    up = BootstrapFilter(pomdp, 500)

    function BetaZero.input_representation(b::ParticleCollection{LTState})
        # Function to get belief representation as input to neural network.
        μ, σ = mean_and_std([s.target for s in particles(b)])
        return Float32[μ..., σ...]
    end

    function get_MGF_encoding(b)
    
        num_Vs = 10
        Vs = [
            (-0.9401886105046561, -1.2640469023502694), 
            (-2.254441586633666, -0.3054737146621892), 
            (0.30156637362537075, 1.4861455416142204), 
            (0.3970965582454251, 0.13307495716011766), 
            (-0.27244135261298613, -0.025061343835141053), 
            (0.6741003709427301, 2.810752246622784), 
            (0.5312379051741923, 0.4042107751203038), 
            (-0.3492096838111564, 1.1150133867980474), 
            (0.28955610981611934, -0.806222078397677), 
            (-0.8400402707094606, -1.3617349037644155)
            ]

        MGF_encoding = Float32[ zeros(num_Vs)... ]
        # m = mean(p.y for p in particles(b)) 
        # MGF_encoding[1] = m
        for i in 1:num_Vs
            v = Vs[i]
            sum_nums = 0.0
            for (s,w) in weighted_particles(b)
                p = s.target
                # sum += w*exp(v*(p-m)/10.0)
                sum_nums += w*exp(sum( Tuple(v.*p)) )
            end
            MGF_encoding[i] = log(sum_nums) 
        end
    
        return MGF_encoding
    end
    
    function BetaZero.input_representation(b::ParticleCollection{LTState})
        # Function to get belief representation as input to neural network.
        # println(b)
        # println("***************************************************")
        # println(particles(b))
        # println("***************************************************")
        # println(weighted_particles(b))
        tbr = get_MGF_encoding(b)
        return tbr
    end

    function BetaZero.accuracy(pomdp::LaserTagPOMDP, b0, s0, states, actions, returns)
        # Function to determine accuracy of agent's final decision.
        return returns[end] == 100.0
    end

    function POMDPs.isterminal(pomdp::POMDP, b::ParticleCollection{T}) where T
        return all(isterminal(pomdp, s) for s in support(b))
    end

    solver = BetaZeroSolver(pomdp=pomdp,
                            updater=up,
                            params=BetaZeroParameters(
                                n_iterations=1,
                                n_data_gen=50,
                            ),
                            nn_params=BetaZeroNetworkParameters(
                                pomdp, up;
                                training_epochs=50,
                                n_samples=100_000,
                                batchsize=1024,
                                learning_rate=1e-4,
                                λ_regularization=1e-5,
                                use_dropout=true,
                                p_dropout=0.2,
                            ),
                            verbose=true,
                            collect_metrics=true,
                            plot_incremental_data_gen=true)

    policy = solve(solver, pomdp)
else
    addprocs(2)
    @everywhere begin
        using BetaZero
        using DMUStudent.HW6
    
        pomdp = LaserTagPOMDP()
        up = BootstrapFilter(pomdp, 500)

        function BetaZero.input_representation(b::ParticleCollection{LTState})
            # Function to get belief representation as input to neural network.
            μ, σ = mean_and_std([s.target for s in particles(b)])
            return Float32[μ..., σ...]
        end

        function BetaZero.accuracy(pomdp::LaserTagPOMDP, b0, s0, states, actions, returns)
            # Function to determine accuracy of agent's final decision.
            return returns[end] == 100.0
        end

        function POMDPs.isterminal(pomdp::POMDP, b::ParticleCollection{T}) where T
            return all(isterminal(pomdp, s) for s in support(b))
        end
    end
    
    solver = BetaZeroSolver(pomdp=pomdp,
                            updater=up,
                            params=BetaZeroParameters(
                                n_iterations=50,
                                n_data_gen=500, # Note increased to 500 when running in parallel.
                            ),
                            nn_params=BetaZeroNetworkParameters(
                                pomdp, up;
                                training_epochs=50,
                                n_samples=100_000,
                                batchsize=1024,
                                learning_rate=1e-4,
                                λ_regularization=1e-5,
                                use_dropout=true,
                                p_dropout=0.2,
                            ),
                            verbose=true,
                            collect_metrics=true,
                            plot_incremental_data_gen=true)
    
    policy = solve(solver, pomdp)
end

#=
Tune the NN

BetaZero.tune_network_parameters(pomdp,solver)

=#


#=
Example of a belief

ParticleCollection{LTState}(LTState[LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [9, 4]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [10, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6]), LTState([5, 1], [6, 1], [8, 5]), LTState([5, 1], [6, 1], [9, 6])], nothing)

=#