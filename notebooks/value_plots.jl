### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2a972d56-b232-11ed-0f98-b16501aaff21
begin
	using Revise
    using Flux
    using Parameters
    using ParticleFilters
    using POMDPModels
    using POMDPs
    using POMDPTools
    using Plots; default(fontfamily="Computer Modern", framestyle=:box)
    using Random
    using Statistics
    using StatsBase
	using Distributions
	using NNlib
	using MCTS
	using DataStructures
	using GaussianProcesses
	using LinearAlgebra
	using GridInterpolations
	using LocalApproximationValueIteration
	using LocalFunctionApproximation
    using Pkg
end

# ╔═╡ 75017254-95fe-4958-a3d2-1ea0b9b8f0a2
begin
	pkg"dev ../"
	using BetaZero
end

# ╔═╡ e38aac74-92a7-4ae7-97c9-57ec6dc9afcd
begin
	pkg"dev ../models/LightDark/"
	using LightDark
	pkg"dev ../models/ParticleBeliefs/"
	using ParticleBeliefs
end

# ╔═╡ 5a5c8125-d8ed-4664-b5a2-a20819a81920
using PlutoUI

# ╔═╡ 2a06238b-06ca-453d-a797-162f183cd869
using Images

# ╔═╡ bda2e95f-b22f-4ba1-95ba-f973669e31d9
using BSON

# ╔═╡ f2e9c419-d2ca-4bc8-8f94-2115f192decf
using Interpolations

# ╔═╡ 0255a5cc-7a77-42dd-b86e-b9e1d2c88079
using Optim

# ╔═╡ 40b78b0d-ddd0-4880-92ba-6bb6c5cb1668
HTML("""
<!-- the wrapper span -->
<div>
	<button id="myrestart" href="#">Restart</button>
	
	<script>
		const div = currentScript.parentElement
		const button = div.querySelector("button#myrestart")
		console.log(button);
		button.onclick = function() { restart_nb() };
		console.log(div.dispatchEvent)
		function restart_nb() {
			console.log("Send event");
			editor_state.notebook.process_status = "no_process";
			window.dispatchEvent(
				new CustomEvent("restart_process"),
				{},
				{ notebook_id: editor_state.notebook.id }
			);
		};
	</script>
</div>
""")

# ╔═╡ 7e65e65a-6cf0-4e3b-afc0-961aeddde9f9
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 48ad6bac-68da-4742-a789-2e5e5b8eed10
LightDarkBetaZero = ingredients("../scripts/representation_lightdark.jl")

# ╔═╡ 0c73c232-95a7-4883-811f-614e676e142a
lightdark_belief_reward = LightDarkBetaZero.lightdark_belief_reward

# ╔═╡ 83e53386-1b8f-4308-863a-fe6fcfa933b6
begin
	Core.eval(Main, :(using BetaZero))
	Core.eval(Main, :(using NNlib))
	Core.eval(Main, :(using Flux))
	Core.eval(Main, :(using LightDark))
	Core.eval(Main, :(using ParticleBeliefs))
	Core.eval(Main, :(include("../scripts/representation_lightdark.jl")))	
	Core.eval(Main, :(using MCTS))
	Core.eval(Main, :(using Random))
	Core.eval(Main, :(using DataStructures))
	Core.eval(Main, :(using ParticleFilters))
	Core.eval(Main, :(using LinearAlgebra))
	Core.eval(Main, :(using GaussianProcesses))
	Core.eval(Main, :(using GaussianProcesses.PDMats))
	Core.eval(Main, :(using GridInterpolations))
	Core.eval(Main, :(using LocalApproximationValueIteration))
	Core.eval(Main, :(using LocalFunctionApproximation))
end

# ╔═╡ c0f29e13-bdce-4908-9e9a-6001c88e5ed9
# policy = BetaZero.load_policy("policy_lightdark_pluto5.bson");
# policy = BetaZero.load_policy("policy_lightdark_pluto_nn_do0.75.bson");
# policy = BetaZero.load_policy("policy_lightdark_pluto_nn_reginputs.bson");
# policy = BetaZero.load_policy("policy_lightdark_pluto_nn_2buffer_5000epochs.bson");
# policy = BetaZero.load_policy("policy_lightdark_pluto_nn_lavi_nice_Psa.bson");
# policy = BetaZero.load_policy("policy_lightdark_pluto_nn_lavi_weekend.bson");
# policy = BetaZero.load_policy("policy_lightdark_pluto_nn_weekend_50iters_include_missing.bson");
# policy = BetaZero.load_policy("policy_lightdark_pluto_nn_weekend_20iters_mse_lr0.001.bson");
policy = BetaZero.load_policy("policy_lightdark_pluto_nn_cleanup.bson");

# ╔═╡ 1bb8df23-575a-479b-a4ce-3ceb3eb7acbf
# solver = BetaZero.load_solver("solver_lightdark_pluto_nn.bson");
# solver = BetaZero.load_solver("solver_lightdark_pluto_gp_1iter.bson");
# solver = BetaZero.load_solver("solver_lightdark_pluto_nn_weekend_50iters_include_missing.bson");

# ╔═╡ 8c7e2291-b846-4659-a3bd-f6abacdd3130
# policy64 = BetaZero.load_policy("policy__lightdark_pluto_nn_thursday_64relu.bson");

# ╔═╡ 6309e3f0-29c9-4963-a2d1-f6fcb5f3ada4
# policy_gp = BetaZero.load_policy("policy_lightdark_pluto_gp_1iter.bson");
# policy_gp = BetaZero.load_policy("policy_gp_regression.bson")
# policy_gp = BetaZero.load_policy("policy_lightdark_pluto_gp_2buffer_10ll.bson")
# policy_gp = BetaZero.load_policy("policy_lightdark_pluto_gp_2buffer.bson")
# policy_gp = BetaZero.load_policy("policy_lightdark_pluto_gp_musig.bson")
# policy_gp = BetaZero.load_policy("policy_lightdark_pluto_gp_lavi.bson");

# ╔═╡ 80c2be9e-a242-4e9d-a082-8c61cfe5ae74
# policy_gp_a = BetaZero.load_policy("policy_lightdark_pluto_gp_actions_20iters.bson");

# ╔═╡ 3af7e1d4-5664-4e71-a075-5695e649bbfa
en = BetaZero.BSON.load("ensemble_m5_weekend_50iters_include_missing.bson")[:en];

# ╔═╡ 243f9939-0d54-457d-be7d-6f0648f4fa63
begin
	lavi_policy = BSON.load(joinpath(@__DIR__, "..", "policy_lavi.bson"))[:policy]
end

# ╔═╡ db354fe2-6fe4-4593-8c8b-9c6bbf6f05e7
md"""
# 1D value plots
"""

# ╔═╡ fb09d547-377a-4e34-9e5f-87d06a8c12e2
@bind stdev Slider(0:0.1:10, default=0, show_value=true)

# ╔═╡ 41d937dd-058b-469d-bc1f-7b70e26bf0c4
@bind skew Slider(-30:0.1:30, default=0, show_value=true)

# ╔═╡ 87c160e9-63f2-4628-b4aa-187e1b85b46d
@bind kurt Slider(0:0.1:30, default=0, show_value=true)

# ╔═╡ 51afcc23-64d6-4da2-91aa-8fe81d01440b
@bind obs Slider(-30:0.1:30, default=0, show_value=true)

# ╔═╡ 4408f005-e0bb-4f96-9f7e-38b1d61c7c9b
begin
	# NOTE: value_plot_1d
	vp_nn = BetaZero.value_plot(policy; σ=stdev, s=skew, k=kurt, o=obs)
	title!("NN")
	vp_gp = BetaZero.value_plot(policy_gp; σ=stdev, s=skew, k=kurt, o=obs)
	title!("GP")
	plot(vp_nn, vp_gp, layout=[1 1], size=(700,250), margin=2Plots.mm)
end

# ╔═╡ 90c4edd0-0215-47df-ba5b-c645576a69f2
md"""
# 2D value & policy plots
"""

# ╔═╡ 07aa5bc1-0a89-4de5-aba7-b74265b7bbed
as = [-1, 0, 1]

# ╔═╡ 03ff8055-8725-4935-8a09-cb1ea6bf6e35
begin
	Σ = 0:0.05:5
	M = -20:0.2:20
	# NOTE x-y flip.
	# Y = (y,x)->policy.surrogate(Float32.([x y fixed_skew fixed_kurt fixed_obs])')[1]
	Y = (y,x)->policy.surrogate(Float32.([x y])')[1]
	Yπ = (y,x)->as[argmax(policy.surrogate(Float32.([x y])')[2:end])]
	# Yπ = (y,x)->rand(SparseCat(as, policy.surrogate(Float32.([x y])')[2:end]))
	Ydata = [Y(x,y) for y in M, x in Σ]
end;

# ╔═╡ 42de9867-be47-44f4-80ec-15a066a45b8c
# TODO: Fix confusing (x,y) (y,x) business

# ╔═╡ 1208652f-3ab9-4379-b0ba-b80f29977b3c
begin
	Σ_small = 0:0.5:5
	M_small = -20:1:20
end;

# ╔═╡ ac5456ea-4211-480f-90c8-b50b4a28d0b9
M_small30 = -30:1:30;

# ╔═╡ 769e3159-bf5e-4294-b7b9-9e49a32406a1
# NOTE x-y flip.
Y_gp = (y,x)->policy_gp.surrogate(Float64.([x y])')[1]

# ╔═╡ eb6537de-b4d2-49bd-9a20-529240fa4e98


# ╔═╡ f105a961-d94e-4c3d-b1d2-6dd077922dc7
exp(0.9162907318741551)^2

# ╔═╡ 812bd7a7-70af-4581-9170-8ce8af05bd14
function shifted_colormap(X; kwargs...)
	xmin, xmax = minimum(X), maximum(X)
	return shifted_colormap(xmin, xmax; kwargs...)
end

# ╔═╡ ae9322da-f369-4f36-9e92-861a1b0f2c4c
function shifted_colormap(xmin, xmax; colors=[:red, :white, :green], rev=false)
	if xmin ≥ 0
		buckets = [0, xmin, xmin/2, xmax/2, xmax] # only non-negatives, anchor at 0
		colors = colors[2:end]
	else
		buckets = [xmin, xmin/2, 0, xmax/2, xmax] # shift colormap so 0 is at center
	end
    normed = (buckets .- xmin) / (xmax - xmin)
    return cgrad(colors, normed, rev=rev)
end

# ╔═╡ ebf5d5bd-6f0e-45ed-af7b-fa06c3140128
fixed_skew, fixed_kurt = 0, 0

# ╔═╡ c84e0d55-33e3-415e-b108-51efd3a13717
@bind fixed_obs Slider(-30:0.1:30, default=0, show_value=true)

# ╔═╡ 848d0d9c-f036-44a1-8ce4-ba12798cd538
function normalize_range(X, a, b)
	xmin = minimum(X)
	xmax = maximum(X)
	return (b - a) * (X .- xmin) / (xmax - xmin) .+ a
end

# ╔═╡ 9b4afdca-9c87-4e88-8199-7fa17c2403fe
begin
	# Ydata′ = Ydata; # clamp.(Ydata, -100, 100); # normalize_range(Ydata, -100, 100);
	Y_flip = (x,y)->policy.surrogate(Float32.([x y])')[1]
	# Ydata′ = deepcopy(Ydata)
	Ydata′ = [Y_flip(y,x) for y in M, x in Σ] # NOTE: trea
	# Ydata′[Ydata′ .< -100] .= 0
	# Ydata′[Ydata′ .> 100] .= 0
end;

# ╔═╡ 029202f4-ba86-457a-911e-30f02ef99f47
policy.surrogate(Float32.([10 3])')

# ╔═╡ 547c9e31-6372-4625-b1a7-212adb2e5438
[2 0]'

# ╔═╡ c6c3bcb9-ecd0-494b-9fa5-11c35c722396
md"""
## Neural network
"""

# ╔═╡ 3bbc44ce-b0b2-4c81-8ec9-d4819bd8ba49
heatmap(Σ, M, Ydata′, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="value (NN)", cmap=shifted_colormap(Ydata′))

# ╔═╡ bd73ffa0-3a73-4922-b7a8-3ffc62f8da25
begin
	Yen_μ = (y,x)->en(Float32.([x y])')[1][1]
	Yen_data = [Yen_μ(x,y) for y in M, x in Σ]
end;

# ╔═╡ ee234170-f86b-4b03-9256-7cbdc644d74a
begin
	Yen_σ = (y,x)->en(Float32.([x y])')[2][1]
	Yen_σ_data = [Yen_σ(x,y) for y in M, x in Σ]
end;

# ╔═╡ 1f52fb7c-56e2-4fc4-9c00-b35b523ac51f
begin
	Yen_data_lcb = Yen_data .- 1.0 * Yen_σ_data
end;

# ╔═╡ 56e2593e-cdf0-4cde-830c-09af3daa7e17
heatmap(Σ, M, Yen_data, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="value (ensemble μ m=5)", cmap=shifted_colormap(Yen_data))

# ╔═╡ 8689d008-0a28-45c1-b35c-0f17c990cb5c
heatmap(Σ, M, Yen_σ_data, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="value (ensemble σ m=5)", cmap=:jet)

# ╔═╡ 30e52cf5-37d4-4b85-908c-43990d6592bd
Yπ_en_uncertain_μ = (y,x)->begin
	Y = en(Float32.([x y])')
	Vμ = Y[1][1]
	Vσ = Y[2][1]
	Pμ = Y[1][2:end]
	if Vσ > 100
	# if Vσ > Vμ
		return 0
	else
		return Vμ
	end
end

# ╔═╡ be289f06-ea10-43d8-9c03-0a14aa1579a4
Yen_μ_data_uncertain = [Yπ_en_uncertain_μ(x,y) for y in M, x in Σ];

# ╔═╡ 5565d9e9-bb35-4bb0-91ad-73662e9d6136
heatmap(Σ, M, Yen_μ_data_uncertain, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="value (ensemble μ m=5)", cmap=shifted_colormap(Yen_μ_data_uncertain))

# ╔═╡ 4e8af1b8-7b5f-442d-b28b-0b8d7bdf1a7d
heatmap(Σ, M, Yen_data_lcb, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="value (ensemble LCB m=5)", cmap=shifted_colormap(Yen_data_lcb))

# ╔═╡ ec606b95-cb6c-4a54-8e1d-d1a1881457d9
begin
	Yen_π_σ = (y,x)->as[argmax(en(Float32.([x y])')[1][2:end])]
end;

# ╔═╡ d75b9ea2-6a15-4e2d-ab78-0eb16ea54b11
heatmap(Σ, M, Yen_π_σ, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (ensemble μ m=5)", cmap=palette(:viridis, 3))

# ╔═╡ c6aa050f-646e-4033-98ef-426c774b2bc9
Yπ_en_uncertain = (y,x)->begin
	Y = en(Float32.([x y])')
	Vμ = Y[1][1]
	Vσ = Y[2][1]
	Pμ = Y[1][2:end]
	if Vσ > 100
	# if Vσ > Vμ
		return -2
	else
		return as[argmax(Pμ)]
	end
end

# ╔═╡ c59a36ee-11e6-420c-9c27-16cbe23cd6e6
heatmap(Σ, M, Yπ_en_uncertain, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (ensemble μ m=5)", cmap=cgrad([:white, :darkblue, :green, :gold], categorical=true)) # palette(:viridis, 3))

# ╔═╡ 157e9d3c-3a4d-4426-86f3-1d731a8da122
begin
	# __μ = 0
	__σ = 2.0
end;

# ╔═╡ fd432cf8-1e7e-475b-9c41-46d18ffe2335
@bind __μ Slider(-10:1:10, show_value=true, default=0)

# ╔═╡ 3b2cffc2-1f52-453e-a244-a3603c829564
Yμ, Yσ = en(Float32.([__μ __σ])')

# ╔═╡ 007c03ca-8a6e-414f-89bd-7fa29fdbf540
en(Float32.([__μ __σ])'; use_lcb=true)

# ╔═╡ 3aae1c42-4092-425f-8086-cbe0518a2692
Plcb = softmax(Yμ[2:end] - 1.0*Yσ[2:end]); # policy LCB

# ╔═╡ 1af2cbfe-4ccf-4b4e-a253-ab6b63d037f1
as[argmax(Plcb)], as[argmax(Yμ[2:end])]

# ╔═╡ 609d9262-10b9-4e9e-9bda-5ad50338ac62
SparseCat(reverse(as), reverse(Plcb)) # LCB P(b,a)

# ╔═╡ 8ab1e3f5-710c-4ecf-9cc8-c1e2d914a011
SparseCat(reverse(as), reverse(Yμ[2:end])) # MEAN P(b,a)

# ╔═╡ ac56db3d-ea72-459e-a072-920edd03cb67
SparseCat(reverse(as), reverse(Yσ[2:end])) # UNCERTAINTY on P(b,a)

# ╔═╡ 40b599eb-e068-4faf-b54e-6dca4896d4a2
Yμ[1] - 1.0*Yσ[1]

# ╔═╡ 4c81fcc5-b50b-4ecd-a0f4-f3ddfe3fad19
SparseCat(reverse(as), reverse(policy.surrogate(Float32.([__μ __σ])')[2:end]))

# ╔═╡ 1515aa19-f8b5-4d26-93a8-3afd3e144ad1
maximum(policy.surrogate(Float32.([-10 0])')[2:end]) ≈ 1

# ╔═╡ 70b0c991-ca46-4434-ac21-4f7785755cb4
Yπ_uncertain = (y,x)->begin
	P = policy.surrogate(Float32.([x y])')[2:end]
	if maximum(P) ≈ 1
		return -2
	else
		return as[argmax(P)]
	end
end

# ╔═╡ 828477f3-d19b-49c2-9827-2cc2d86233e6
md"""
## BetaZero (online)
"""

# ╔═╡ 47fb1fc3-dbeb-4696-ae7a-d0f049c3ccf3
md"""
TODO: policy plot within BetaZero (not raw network)
"""

# ╔═╡ 7ef35aa9-8558-40aa-abee-1f3b671004c0
length(Σ_small)*length(M_small)

# ╔═╡ 35c24d73-3c31-4d8b-9ab3-3acc359911d7
length(Σ)*length(M)

# ╔═╡ bfa256df-3c81-4d55-8384-aa277ea3a645


# ╔═╡ 98b53fd7-d853-4869-849b-e6875ab89c5d
# heatmap(Σ, M, Yπ_online, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online)", cmap=palette(:viridis, 3))

# ╔═╡ 9c523eb3-94ce-4fe6-a241-b9fa7d1df25b
# convert_s(ParticleHistoryBelief, [0, 0], solver.bmdp)

# ╔═╡ 05d7ecc1-34c4-4b34-bd3b-4cb3ad0267b7
function POMDPs.value(policy::BetaZeroPolicy, b)
	a, info = action_info(policy, b; tree_in_info=true)
	tree = info[:tree]
	q = tree.q
	return q[1] # root node value
end

# ╔═╡ f60f4c7a-9b4b-48c2-921b-0fa4214ba435
value_online = (x,y,policy) -> begin
	Random.seed!(0)
	value(policy, convert_s(ParticleHistoryBelief, [x, y], solver.bmdp))
end

# ╔═╡ bc965027-5da1-4377-9ed5-e72b9452d526
policy.planner.solver.n_iterations = 10; # NOTE.

# ╔═╡ 54b15814-9d64-4799-bdfb-514e2a3e48bb
# ╠═╡ disabled = true
#=╠═╡
value_online_large = [value_online(y,x,en_policy) for y in M, x in Σ];
  ╠═╡ =#

# ╔═╡ e6732dc1-ab93-4d61-b215-168cbb00c0ee
#=╠═╡
heatmap(Σ, M, value_online_large, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online)", cmap=shifted_colormap(value_online_large))
  ╠═╡ =#

# ╔═╡ 86f1ed3f-d58a-44bf-ba6c-dbb2b04b4e8c
heatmap(Σ, M, Yπ, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (NN)", cmap=palette(:viridis, 3))

# ╔═╡ 64f10375-ad0b-4640-b117-fe96bf1ea9e1
heatmap(Σ, M, Yπ_uncertain, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (NN)", cmap=cgrad([:white, :darkblue, :green, :gold], categorical=true)) # palette(:viridis, 3))

# ╔═╡ 0dea654f-e4de-4ecf-8ad4-ca2ec76136c9
Yπ64 = (y,x)->as[argmax(policy64.surrogate(Float32.([x y])')[2:end])]

# ╔═╡ 2bd13b69-31fc-4885-a5f5-d95457873445
heatmap(Σ, M, Yπ64, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (64-NN)", cmap=palette(:viridis, 3))

# ╔═╡ 8c6f978c-afe6-481e-a52d-63fb597737d5
policy64.planner.solver.n_iterations = 10; # NOTE for 'online' results

# ╔═╡ b0caab16-2b08-4683-b14f-ba3bcf3275ef
value_online_small64 = [value_online(y,x,policy64) for y in M_small, x in Σ_small];

# ╔═╡ e84ca904-a339-4d43-a3ae-41988868ecee
heatmap(Σ_small, M_small, value_online_small64, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online [64])", cmap=shifted_colormap(value_online_small64))

# ╔═╡ f278d28d-ccd0-411a-a74e-fbaa4efca965
begin
	value_online_64 = [value_online(y,x,policy64) for y in M, x in Σ];
	heatmap(Σ, M, value_online_64, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online [64])", cmap=shifted_colormap(value_online_64))
end

# ╔═╡ df1ca98b-68a2-413d-9253-fe709d463ce9
begin
	Yπ_online64 = (y,x)->action(policy64, convert_s(ParticleHistoryBelief, [x, y], solver.bmdp))
	heatmap(Σ_small, M_small, Yπ_online64, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online [64])", cmap=palette(:viridis, 3))
end

# ╔═╡ f9f24938-4e9e-46bf-926a-646c937e48e2
heatmap(Σ, M, Yπ_online64, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online [64])", cmap=palette(:viridis, 3))

# ╔═╡ c5591821-7615-4351-9a01-12c82efb746f
md"""
## RGB policy plot
"""

# ╔═╡ 8b172603-92f0-4750-823e-02fc61eeaa6b
Yπ_rgb = (y,x)->as[argmax(policy.surrogate(Float32.([x y])')[2:end])]

# ╔═╡ 1d309991-1c38-40e3-874a-884dc83709da
# heatmap(Σ, M, Yπ_rgb, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy distribution (NN)", c=policy_rbgs)

# ╔═╡ 5658517d-5b0e-4a58-982d-e7cb08b3522e
md"""
| | | |
| --- | --- | --- |
| `red (-1)` | `green (0)` | `blue (+1)` |
"""

# ╔═╡ 0feeb06c-3f6d-4ec7-b795-9a1c52fc0e69
0.90^20 * 100 # e.g., up +1x10, down +1x10, stop action 0

# ╔═╡ 11d10167-0a74-4153-b618-52ae07889324
md"""
## Gaussian procces
"""

# ╔═╡ 4f5e76f2-97d3-418e-a1e5-2fc558fbddcb
heatmap(Σ, M, Y_gp, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="value (GP)", cmap=shifted_colormap([Y_gp(x,y) for x in Σ for y in M]))

# ╔═╡ 7f8ae7e3-8144-4d38-b398-edecf5fc396f
@bind fixed_action Slider([-1, 0, 1], default=0, show_value=true)

# ╔═╡ 42b011d7-268b-4c9f-94d9-bbc4981847c1
# NOTE x-y flip.
Y_gp_a = (y,x)->policy_gp_a.surrogate(Float64.([x y fixed_skew fixed_kurt fixed_action fixed_obs])')[1]

# ╔═╡ 607ed6b8-af59-402a-b9ba-6786bb3f248c
# heatmap(Σ, M, Y_gp_a, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="value (GP)", cmap=shifted_colormap([Y_gp_a(x,y) for x in Σ for y in M]))

# ╔═╡ 1bea8485-6ccd-4108-a97d-9669f6669da8
md"""
# Light-dark visualizations
"""

# ╔═╡ 14ea3b14-a718-43de-a126-7910fb4e02c1
PlotLightDark = ingredients("../scripts/plot_lightdark.jl")

# ╔═╡ 2f598533-6de2-4029-b3f9-7a2904c8d766
plot_lightdark = PlotLightDark.plot_lightdark;

# ╔═╡ f98dd3af-2af8-40b1-b71e-d8e842b411c3
begin
	en_policy = BetaZero.attach_surrogate!(deepcopy(policy), en)
	en_policy.planner.solver.n_iterations = 10; # NOTE.
end;

# ╔═╡ 598bf115-667e-49e5-bdad-bcef4e3e1cbc
Yπ_online = (y,x)->action(en_policy, convert_s(ParticleHistoryBelief, [x, y], solver.bmdp))

# ╔═╡ 63364798-48c4-4a7b-8f91-29bda69307f1
heatmap(Σ_small, M_small, Yπ_online, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online)", cmap=palette(:viridis, 3))

# ╔═╡ 1e5d06fe-7036-4a13-83ec-7cf301573c25
Random.seed!(0); value(en_policy, convert_s(ParticleHistoryBelief, [0, 1], solver.bmdp))

# ╔═╡ bb851367-0421-4e36-b20b-d11fc57874b5
value_online_small = [value_online(y,x,en_policy) for y in M_small, x in Σ_small];

# ╔═╡ ff8d9df0-7635-4f9e-a3ee-c279977f8173
heatmap(Σ_small, M_small, value_online_small, xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online)", cmap=shifted_colormap(value_online_small))

# ╔═╡ cd4dab5b-a7cd-4e07-b6aa-98d1220ad0c8
policy_rbgs = [RGB(en_policy.surrogate(Float32.([y x])')[2:end]...) for x in Σ, y in M];

# ╔═╡ 38e1c560-beaf-4c2e-8f5c-d6b08a4d3273
plot(imresize(rotl90(policy_rbgs), 400, 600))

# ╔═╡ 74f2b6ec-4e5a-4887-8e31-3cd41d54b375
shared_seed = 1;

# ╔═╡ 6d822d41-f49a-4d11-8266-e41a02f5aaa1
# NOTE change to initial state distribution
POMDPs.initialstate(pomdp::LightDarkPOMDP) = initialstate(pomdp; isuniform=true)

# ╔═╡ 2acb5d21-e5b4-446c-80d0-ee98f3bbd4a7
# plot_lightdark(pomdp_cost, en_policy, up; max_steps=20, seed=shared_seed)

# ╔═╡ dd9e1fcc-79c1-4452-b032-51f918303f17
# plot_lightdark(pomdp_cost, policy, up; max_steps=100, seed=shared_seed)

# ╔═╡ 73e8ca7f-cf0a-431f-8625-7e317224bed4
begin
	radius = 60
	ps = ones(2*div(radius,2)+1)
    ps /= length(ps)
    rand(SparseCat(div(-radius,2):div(radius,2), ps), 1000000) |> histogram
end

# ╔═╡ 6f7bc73f-3961-45af-90ff-002e02dbe9eb
md"""
# Sims
"""

# ╔═╡ 9186d51c-dac7-4b21-b4eb-3cd724bb1a3e
# ╠═╡ disabled = true
#=╠═╡
data = BetaZero.sample_data(solver.data_buffer_valid, 1000)
  ╠═╡ =#

# ╔═╡ ce006bd6-d691-4efb-b740-4ddc258416c4
pomdp = solver.pomdp;

# ╔═╡ ccf70808-5103-4279-846d-237f888fba01
begin
	pomdp_cost = deepcopy(pomdp)
	pomdp_cost.movement_cost = 0.1
end;

# ╔═╡ e20fbafd-0f76-4862-a1b3-a7fff41b47b1
[s.y for s in rand(initialstate(pomdp; isuniform=true), 1000000)] |> histogram

# ╔═╡ ad45a730-5950-43b6-ba6b-82ff9373c59d
[s.y for s in rand(initialstate(pomdp; isuniform=false), 1000000)] |> histogram

# ╔═╡ f97f9786-5cc7-478f-997c-2fb26053a504
begin
	solver.belief_reward = (pomdp, b, a, bp) -> mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))
	BetaZero.fill_bmdp!(pomdp, solver)
	policy_gp2 = BetaZero.solve_planner!(solver, policy_gp_a.surrogate)
end;

# ╔═╡ d1296cde-fe14-4ca0-84bf-540c7752125e
up = solver.updater;

# ╔═╡ a2f127dc-d372-40ec-851b-06022890710f
ds0 = initialstate(pomdp)

# ╔═╡ 451a9c77-f76e-4912-bb3e-5c02a24b1b84
# Base.convert(::Type{Main.ParticleHistoryBelief}, b::ParticleHistoryBelief) = Main.ParticleHistoryBelief(b.particles, b.observations, b.actions)

# ╔═╡ 1709523a-8674-43a7-94c0-de9fc9a5479a
b0 = initialize_belief(up, ds0)

# ╔═╡ 898c7dfe-fc02-4810-8d37-4b06247feea9
π = RandomPolicy(pomdp; updater=up);

# ╔═╡ d2955613-e976-4b5b-81ba-1c9454ec2c03
ParticleFilters.support(b::ParticleHistoryBelief) = particles(b)

# ╔═╡ fdccfa1f-da99-4b59-b647-a426ecd7566a
# ParticleFilters.support(b::Main.ParticleHistoryBelief) = particles(b)

# ╔═╡ 652b3290-815a-4e33-80c3-3d7ef84bd8af
action(policy_gp2, b0)

# ╔═╡ 5853347f-86e7-459d-83cd-ba23239ffb98
policy2 = BetaZeroPolicy(policy.surrogate, solve(solver.mcts_solver, solver.bmdp));

# ╔═╡ 4b0b6f23-1fca-4907-bdb3-eafb3bbc9606
begin
	n = 10
	expectation = []
	predicted = []
	for i in 1:n
		b0 = initialize_belief(up, ds0)
		s0 = rand(b0)
		𝔼G = [simulate(RolloutSimulator(max_steps=200), pomdp, policy_gp2, up, b0, s0) for _ in 1:10]
		Ṽ = policy_gp2.surrogate(BetaZero.input_representation(b0))[1]
		push!(expectation, mean(𝔼G))
		push!(predicted, Ṽ)
	end
end

# ╔═╡ 6f2b96cc-c0cf-42fa-8cec-380ff48d9106
𝔼G = [simulate(RolloutSimulator(max_steps=200), pomdp, π, up, b0, rand(b0)) for _ in 1:100]

# ╔═╡ 6213273f-0417-4cbc-9eff-9add06948401
begin
	histogram(predicted, label="predicted")
	histogram!(expectation, label="𝔼", alpha=0.5)
end

# ╔═╡ 8d422b96-b8ef-4d36-ab80-41d404cf4ff0
function plot_bias_2(model, data)
    Plots.plot(size=(500,500), ratio=1, xlims=(-100, 100), ylims=(-100, 100))
    Plots.xlabel!("true value")
    Plots.ylabel!("predicted value")
    Plots.scatter!(vec(data), vec(model), c=:MediumSeaGreen, label=false, alpha=0.5)
    Plots.abline!(1, 0, label=false, c=:black, lw=2)
end

# ╔═╡ 343bdb4e-f9ba-4c10-a711-750d695652f7
plot_bias_2(predicted, expectation)

# ╔═╡ 6572e23c-d4f4-4803-8034-6613fdf18960
md"""
# Gaussian process regression
"""

# ╔═╡ 47057cbd-508c-48d8-a2aa-59f3a5296481
dX = solver.data_buffer_train[1].X

# ╔═╡ 2a860403-efcf-40d9-81a0-bd0f2b76081a
dYV = solver.data_buffer_train[1].Y[1,:]

# ╔═╡ ac665280-9b69-4eb4-b4ca-e794fb3da01b
histogram2d(dX[2,:], dX[1,:], nbins=100, c=:viridis, show_empty_bins=false)

# ╔═╡ b4a20910-2b1c-4697-9140-eb619fffbe40
dX[2,:]

# ╔═╡ 8dca08c1-e469-44a1-95ad-af4c76ba2729
itp = LinearInterpolation((dX[1,:], dX[2,:]), dYV)

# ╔═╡ b2c7630b-3673-4158-9e91-4fbd2f471d08
function tli()
	x = range(-2, 3, length=20)
	y = range(3, 4, length=10)
	z = @. cos(x) + sin(y')
	# Interpolatant object
	itp = LinearInterpolation((x, y), z)
end

# ╔═╡ 92d0d38f-9fb6-4c08-b569-6b336d8fba8a
heatmap(dX[2,:], dX[1,:], [dyv for dyv in dYV, _ in dYV], cmap=shifted_colormap(dYV))

# ╔═╡ 32c79abb-122c-421a-831e-28b4faaad279


# ╔═╡ c6a3bcf0-c7f4-468a-9884-f7fe4231ab5d
repeat(dYV, outer=2)

# ╔═╡ ce515ed9-bac3-4b48-977f-c94937eebe8f
#=╠═╡
gp_plts[2]
  ╠═╡ =#

# ╔═╡ 42999b4d-7b41-41b8-b719-2e375d890c97
function basis_regression(X, y, ϕ)
    B = mapreduce(x->ϕ(x)', vcat, X)
    θ = B\y
    return x -> ϕ(x)'θ
end

# ╔═╡ 2694d4ba-5eaf-4bf7-a9eb-a9e407163cf1
ϕdegree(x, a=x[1], b=x[2]) =
	[1, 
     a, b,
     a^2, a*b, b^2,
     a^3, a^2*b, a*b^2, b^3,
     a^4, a^3*b, a^2*b^2, a*b^3, b^4,
     a^5, a^4*b, a^3*b^2, a^2*b^3, a*b^4, b^5,
     a^6, a^5*b, a^4*b^2, a^3*b^3, a^2*b^4, a*b^5, b^5]

# ╔═╡ 7b237f3d-6898-4c57-b672-7449f8a485b0
ϕ2(x) = [1, norm(x,2), norm(x.^2, 2), norm(x.^3, 2), norm(x.^4, 2), norm(x.^5, 2), norm(x.^6, 2), norm(x.^7, 2)]

# ╔═╡ bcd84550-5656-4997-a4ce-1167852c7eec
ϕ3(x) = [1, norm(x,Inf), norm(x.^2,Inf), norm(x.^3,Inf), norm(x.^4,Inf), norm(x.^5,Inf), norm(x.^6,Inf), norm(x.^7,Inf)]

# ╔═╡ f7a13b63-f7f5-41cf-bad4-7d8a5f588c6d
ϕ4(x) = [1, sum(x), sum(x.^2), sum(x.^3), sum(x.^4), sum(x.^5), sum(x.^6), sum(x.^7)]

# ╔═╡ c32085e0-8f24-4e87-bedc-d18e8197e575
function run_rb(rb::Vector, b̃)
	return as[argmax(softmax(map(f->f(b̃), rb)))]
end

# ╔═╡ 59ca30e3-7397-4245-8985-c25795e46750
begin
	mt = 1000
	mv = trunc(Int, mt*0.8)
end

# ╔═╡ 8d818d7f-0f98-4131-a723-12fd164d218f
data = BetaZero.sample_data(solver.data_buffer_train, mt);

# ╔═╡ 31684f7c-f5ad-4ea9-993e-cd19c38a8eae
data2 = BetaZero.sample_data(solver.data_buffer_valid, mv);

# ╔═╡ fa79f064-81df-476c-a3fb-5271bee3cd5a
# data = BSON.load("data1.bson")[:data];

# ╔═╡ 410bdb7a-a201-47fb-9a92-9910b5b93ca0
# data2 = BSON.load("data2.bson")[:data2];

# ╔═╡ 8f0506e3-30df-4093-afa1-cb035673cf01
@bind ℓ_slide_P Slider([0.1:0.1:10;], show_value=true)

# ╔═╡ c5332871-a0d3-4586-8be3-643fcd9bb392
@bind σ_slide_P Slider([0.1:0.1:10;], show_value=true)

# ╔═╡ 00a84f88-9d7e-4344-b813-f66b6a70c394
solver.gp_params

# ╔═╡ 6b2514dd-4312-4030-b4b0-d26d6f936c5d
@bind ℓ_slide Slider([0.1:0.1:10;], show_value=true)

# ╔═╡ 46166104-92e8-4f5a-bcd4-35435f18aa63
@bind σ_slide Slider([0.1:0.1:10;], show_value=true)

# ╔═╡ f067942f-86a6-4edc-bcf0-78fecb6455a0
begin
	solver.gp_params.verbose_plot = false
	solver.gp_params.kernel = SE(log(ℓ_slide), log(σ_slide))
	solver.gp_params.n_samples = 100
	f_init = BetaZero.initialize_gaussian_proccess(solver)
	f_train = BetaZero.train(f_init, solver; verbose=false)
end

# ╔═╡ 0cf1d477-d750-4a45-8bd7-7a3c061c6532
heatmap(Σ_small, M_small, (x,y)->as[argmax(f_train.fP([x,y]))], xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online)", cmap=palette(:viridis, 3))

# ╔═╡ b3e3c159-31f8-4d3f-b062-43dc4b940db6


# ╔═╡ 12458ba9-10ed-4ebc-88fd-140f86b79920
solver.gp_params.kernel.σ2

# ╔═╡ a8782a5e-9a51-470b-9d89-c8188c04dc12
results = BetaZero.gp_tune(pomdp, solver, f_init;
	ℓ_range=Distributions.Uniform(0.0001, 50),
	σ_range=Distributions.Uniform(0.0001, 50),
	resources=10000)

# ╔═╡ 93be835a-a3e0-4147-a2ad-3a86d93a34b5
begin
	# gp_value_θ, gp_value_err, fV_tuned = collect(values(results["value"]))[argmin(map(v->v[2], values(results["value"])))]
	gp_value_θ, gp_value_err, fV_tuned = results["value"]
	gp_ỹ_tuned = [fV_tuned([x,y]) for x in M, y in Σ] # NOTE x-y flip
	cmap_model_tuned = shifted_colormap(gp_ỹ_tuned)
	plt_gp_tuned_v = heatmap(Σ, M, gp_ỹ_tuned, c=cmap_model_tuned)
end;

# ╔═╡ 980c2630-b66b-423f-8153-3a979c16bdf6
begin
	# gp_policy_θ, gp_policy_err, fP_tuned = collect(values(results["policy"]))[argmin(map(v->v[2], values(results["policy"])))]
	gp_policy_θ, gp_policy_err, fP_tuned = results["policy"]
	gp_p̃_tuned = [as[argmax(fP_tuned([x,y]))] for x in M, y in Σ] # NOTE x-y flip
	plt_gp_tuned_p = heatmap(Σ, M, gp_p̃_tuned, c=palette(:viridis, 3))
end;

# ╔═╡ 31a99d07-aa61-46e1-827f-05d3023d0a09
plot(plt_gp_tuned_v, plt_gp_tuned_p, layout=2, size=(800,300))

# ╔═╡ 61284855-5286-487a-b6b0-f8a24adb28a3
gp_policy_θ

# ╔═╡ d4d9e5b3-3f6c-4936-9592-8b08ed02a09e
begin
    X = data.X[:,1:mt]'
    X2 = data2.X[:,1:mv]'
    y = data.Y[1:mt]
    y2 = data2.Y[1:mv]
	gp_X = Float64.(X)'
    gp_mean = MeanZero() # MeanConst(Float64(mean(y)))
end; md"$X,y$"

# ╔═╡ 8f196424-1b11-4ee5-bc58-ee5284fdf098
br_predict = basis_regression(collect(eachcol(gp_X)), y, ϕdegree)

# ╔═╡ 894ee284-a417-4017-9696-6f741706f179
begin
	rbPs = []
	num_actions = length(actions(pomdp))
    for i in 1:num_actions
        y_train_Pi = data.Y[i,:]
		fPi = basis_regression(collect(eachcol(gp_X)), y_train_Pi, ϕ2)
        push!(rbPs, fPi)
    end
end

# ╔═╡ dc49a10c-4648-4e8d-bc85-cc6cfc073c5c
run_rb(rbPs, [0,0])

# ╔═╡ 7d7670e3-d07b-4175-b399-68327a1c3b03
heatmap(Σ, M, (x,y)->run_rb(rbPs, [x,y]), xlabel="\$\\sigma(b)\$", ylabel="\$\\mu(b)\$", title="policy (BetaZero online)", cmap=palette(:viridis, 3))

# ╔═╡ 886e5e3c-ec60-4da3-bf49-18e811578853
rbPs

# ╔═╡ 88362011-8e3b-45de-b451-57265459aa71
fgp = gp->xy->predict_f(gp, Float64.(reshape(xy, (:,1))))[1][1] # mean

# ╔═╡ 87d7505d-2edb-401d-ac2e-9386f5ccad5f
σgp = gp->xy->predict_f(gp, Float64.(reshape(xy, (:,1))))[2][1] # std

# ╔═╡ baa3ab4f-be38-4851-a851-4a37b38a3f66
begin
    ν = 1/2

	ll = log(ℓ_slide)
	# ll = log(0.45)

	lσ = log(σ_slide)
	# lσ = log(0.1)

	# kernel = Matern(ν, ll, lσ)
	kernel = SE(ll, lσ)

	gp = GP(gp_X, y, gp_mean, kernel)
	f2 = fgp(gp)
	σ2 = σgp(gp)
end

# ╔═╡ 8d465a2a-2138-4415-92aa-f0c4c20bdd6d
gp.kernel.σ2 = 6.25

# ╔═╡ 5be544ba-d05f-4e4e-b88e-3f7cc1ee79c4
begin
	# gp_lcb_small = [f([x,y]) - σ([x,y]) for x in M, y in Σ] # NOTE x-y flip
	gp_ỹ_small = [f2([x,y]) for x in M_small, y in Σ_small] # NOTE x-y flip
	cmap_model_small = shifted_colormap(gp_ỹ_small)
	plt_gp_mean = heatmap(Σ_small, M_small, gp_ỹ_small, c=cmap_model_small)
	
	gp_std_small = [σ2([x,y]) for x in M_small, y in Σ_small] # NOTE x-y flip
	plt_gp_std = heatmap(Σ_small, M_small, gp_std_small, c=:viridis)

	plot(plt_gp_mean, plt_gp_std, layout=2, size=(700,200))
end

# ╔═╡ e56a6370-00e5-4305-8062-8106dc2df9cb
begin
	gp_error(x, y) = (y - f2(x))^2
	mean_error_train = mean(gp_error(xi, yi) for (xi,yi) in zip(eachrow(X), y))
	mean_error_valid = mean(gp_error(xi, yi) for (xi,yi) in zip(eachrow(X2), y2))
	@info "Mean error (train): $mean_error_train"
	@info "Mean error (validation): $mean_error_valid"
end

# ╔═╡ fc28e360-a595-46e3-ba32-d917ce0b9f63
# ╠═╡ disabled = true
#=╠═╡
begin
	loss_train = Float64[]
	loss_valid = Float64[]
	ℓ_range = range(0.2, 5, length=20)
	σ_range = range(0.1, 50, length=20)
	losses = Matrix{Float64}(undef, length(ℓ_range), length(σ_range))
	for (i,ℓ) in enumerate(ℓ_range)
		for (j,σ) in enumerate(σ_range)
			ll = log(ℓ)
		    lσ = log(σ)
		    kernel = SE(ll, lσ)
		    gp = GP(gp_X, y, gp_mean, kernel)
		    f = xy->predict_f(gp, Float64.(reshape(xy, (:,1))))[1][1]

			loss = (x, y)->(y - f(x))^2
			err_train = mean(loss(xi, yi) for (xi,yi) in zip(eachrow(X), y))
			err_valid = mean(loss(xi, yi) for (xi,yi) in zip(eachrow(X2), y2))
			@info "ℓ=$ℓ, σ=$(σ): $((err_train, err_valid))"
			push!(loss_valid, err_valid)
			push!(loss_train, err_train)
			losses[i,j] = err_valid
		end
	end
end
  ╠═╡ =#

# ╔═╡ 2e512cda-bba3-4cc9-9110-132ffa37e9e5
#=╠═╡
begin
	# heatmap(ℓ_range, σ_range, reshape(loss_valid, (length(ℓ_range),length(σ_range)))')
	heatmap(ℓ_range, σ_range, losses')
	xlabel!("\$\\ell\$")
	ylabel!("\$\\sigma\$")
end
  ╠═╡ =#

# ╔═╡ 1ae1a1e8-aaec-47b2-9ced-6fd8afc48b9e
#=╠═╡
begin
	best = argmin(losses)
	# ℓ = ℓ_range[best.I[1]]
	# σ = σ_range[best.I[2]]
end
  ╠═╡ =#

# ╔═╡ 01f7dcdf-fff2-4b58-93da-585c1cebd193
#=╠═╡
ℓ_range[best.I[1]], σ_range[best.I[2]]
  ╠═╡ =#

# ╔═╡ b9cb080f-087f-4d4b-b575-2190652b92ea
#=╠═╡
begin
	plot(loss_valid, label="valid")
	plot!(loss_train, label="train")
end
  ╠═╡ =#

# ╔═╡ 8f2b6e82-4e3d-440d-ba16-ad043b833b97
normalize01(x, X) = (x - minimum(X)) / (maximum(X) - minimum(X))

# ╔═╡ c455f35d-25d7-44d7-90dd-a2cb9c42020c
begin
	scatter_cmap = shifted_colormap(dYV)
	scatter(dX[2,:], dX[1,:], alpha=0.1, cmap=[get(scatter_cmap, normalize01(yi,dYV)) for yi in dYV])
end

# ╔═╡ 9ee1c3b0-d600-402f-bc42-4fa2feeee807
begin
	x1min, x1max = minimum(X[:,1]), maximum(X[:,1])
	x2min, x2max = minimum(X[:,2]), maximum(X[:,2])
	x1min2, x1max2 = minimum(X2[:,1]), maximum(X2[:,1])
	x2min2, x2max2 = minimum(X2[:,2]), maximum(X2[:,2])
	x1min = min(x1min, x1min2)
	x1max = max(x1max, x1max2)
	x2min = min(x2min, x2min2)
	x2max = max(x2max, x2max2)
	ymin, ymax = minimum(y), maximum(y)
	ymin2, ymax2 = minimum(y2), maximum(y2)

	pltX = x2min:0.01:x2max
	pltY = x1min:0.1:x1max

	# cmap = shifted_colormap(y)
	gp_ỹ = [f([x,y]) for x in pltY, y in pltX] # NOTE x-y flip
	# gp_ỹ = [f([x,y,0,0,0]) for x in pltY, y in pltX] # NOTE x-y flip
	cmap_data = shifted_colormap([min(ymin,ymin2), max(ymax,ymax2)])
	cmap_model = shifted_colormap(gp_ỹ)
end

# ╔═╡ d9acdd13-cb52-4ea3-a493-2534f43e236a
begin
	Xrange = range(0, stop=5, length=100)
	Yrange = range(-10, stop=10, length=100)
	br_Y = [br_predict([x,y]) for x in Yrange, y in Xrange] # NOTE x-y
	heatmap(Xrange, Yrange, br_Y, cmap=shifted_colormap(br_Y))
	scatter!(X[:,2], X[:,1], cmap=[get(cmap_data, normalize01(yi,y)) for yi in y], marker=:square, msc=:gray, alpha=0.5, label="training")
end

# ╔═╡ c361ac8e-51bf-436b-a195-b728ceceb4ce
# ╠═╡ disabled = true
#=╠═╡
begin
	gp_plts = []
	for training in [true, false]
		plot(size=(800,400), legend=:outerbottomleft)
	
		# NOTE: Switch (μ on y-axis, σ on x-axis)
		# heatmap!(pltX, pltY, (y,x)->f([x,y,0,0,0]), c=cmap_model)
		heatmap!(pltX, pltY, gp_ỹ, c=cmap_model)
		training && push!(gp_plts, deepcopy(plot!()))

		if training
			scatter!(X[:,2], X[:,1], cmap=[get(cmap_data, normalize01(yi,y)) for yi in y], marker=:square, msc=:gray, alpha=0.5, label="–training-")
		else
			scatter!(X2[:,2], X2[:,1], cmap=[get(cmap_data, normalize01(yi,y2)) for yi in y2], marker=:circle, msc=:black, alpha=0.5, label="validation")
		end
	
		xlabel!("\$\\sigma(b)\$")
		ylabel!("\$\\mu(b)\$")
		title!(training ? "training" : "validation")
		push!(gp_plts, plot!())
	end
end
  ╠═╡ =#

# ╔═╡ eebb4757-83de-490a-84d5-52f988708c3a
#=╠═╡
gp_plts[1]
  ╠═╡ =#

# ╔═╡ ddbf075e-539e-49a7-8a1b-a8e7b6eb644f
#=╠═╡
gp_plts[2]
  ╠═╡ =#

# ╔═╡ 1fa63234-37e4-4b12-a064-6a55b2ee0b67
#=╠═╡
gp_plts[3]
  ╠═╡ =#

# ╔═╡ 047c2c69-d662-4dba-b8b6-c25167178ecb
md"""
# Plots
"""

# ╔═╡ 0f8578cd-37f5-4ecd-b9eb-6a6351750fa9
begin
	p1 = plot([1], [1], marker=false)
	p2 = plot([1], [1], marker=false)
	p3 = plot([1], [1], marker=false)
	plot(p1, p2, p3, layout=@layout[a{0.5w} b{0.5w}; _ _ _ c{0.5w} _ _ _ _])
end

# ╔═╡ 743cf6ad-ddaa-403f-8ab8-83513c804134
ho_μ = [m.mean for m in solver.holdout_metrics];
# ho_μ = [4.236551205969867, 3.652028459679382, 7.7944189894548614, 10.318113003732424, 10.287302809230992, 11.526361419485301, 12.069473152343456, 11.84934767877954, 8.896239195886608, 10.044561362443707, 10.965512347196158, 10.435503538777873, 7.1999703177673124, 7.129402415779563, 10.84189518533771, 10.900835704243512, 11.63269827863645, 12.132977187490116, 11.389794019740489, 12.749711348500671, 10.38277757430994, 10.595804967910901, 13.398598348450202, 12.701389219548489, 9.295595397652649, 11.292780474121894, 11.140712120404325, 10.973793896874584, 13.351930654269845, 12.04749657661156, 12.782538022214517, 10.921692721641527, 12.664604736899955, 11.719984818431652];

# ╔═╡ 7c7b0c5a-ca81-4906-b0a2-ab0c2f1b7782
ho_σ = [m.std for m in solver.holdout_metrics];
# ho_σ = [32.52126580543532, 11.409984157920357, 13.594869250610236, 12.28528625779923, 12.521762409373638, 9.89231860480156, 15.343509355635812, 12.533694460944085, 14.233244899468765, 7.131522004805157, 13.408475142519846, 13.667394373029472, 13.064810931241205, 18.11261239946177, 13.86379431526749, 7.8652487728454386, 5.484074355145341, 8.431818002213413, 9.877067852736923, 10.32565045807811, 9.910889011516751, 10.97362461662418, 5.087895762661509, 9.183794792257446, 16.521760832568063, 10.830610073878386, 9.055981842492336, 9.91984691155096, 5.417702137482001, 8.05703332380186, 8.251618277655956, 11.131211067924784, 6.7756796175012, 12.095940437591379];

# ╔═╡ 351d5b4f-4347-441a-88ef-06d54eaf0c16
n_ho = solver.params.n_holdout;

# ╔═╡ 058622d9-db81-4a0b-a7a7-dcc5fb765090
ho_stderr = ho_σ ./ sqrt(n_ho);

# ╔═╡ e52988c7-a2d1-411c-a0cf-119180b2d0e0
begin
	ho_accs = [mean_and_std(m.accuracies) for m in solver.holdout_metrics]
	ho_acc_μ = first.(ho_accs)
	ho_acc_σ = last.(ho_accs)
	ho_acc_stderr = ho_acc_σ ./ sqrt(n_ho)
end;

# ╔═╡ fd9ece7a-2466-47b8-94b5-02dd5f3181ca
begin
	pm_returns = BetaZero.returns_over_time(solver, mean_and_std)
	pm_μ = first.(pm_returns)
	pm_σ = last.(pm_returns)
	n_pm = solver.params.n_data_gen;
	pm_stderr = pm_σ ./ sqrt(n_pm)
end;

# ╔═╡ d1977228-b7a2-4a87-b556-c111039509ce
begin
	pm_accs = BetaZero.accuracy_over_time(solver, mean_and_std)
	pm_acc_μ = first.(pm_accs)
	pm_acc_σ = last.(pm_accs)
	pm_acc_stderr = pm_acc_σ ./ sqrt(n_pm)
end;

# ╔═╡ ad0b9cfa-a1e6-479c-99ae-179b778c716b
solver.holdout_metrics

# ╔═╡ daafbdec-0311-4981-a8e2-201edeca5b1e
lavi_results = (12.94, 0.498) # mean, stderr

# ╔═╡ a68e35de-883c-46ec-ab66-6a77b43a04c7
lavi_acc_results = (0.856, 0.011) # mean, stderr

# ╔═╡ 99608de8-ef1c-473c-a8d4-f21d9800e389
function rolling_max(Y, Y2=nothing)
	max_Y = Vector{Float64}(undef, length(Y))
	use_Y2 = !isnothing(Y2)
	if use_Y2
		max_Y2 = Vector{Float64}(undef, length(Y2))
	end
	for i in eachindex(Y)
		if i == 1
			max_Y[i] = Y[i]
			if use_Y2
				max_Y2[i] = Y2[i]
			end
		else
			prev_max_idx = argmax(max_Y[1:i-1])
			prev_max = max_Y[prev_max_idx]
			curr_max_idx = Y[i] > prev_max ? i : prev_max_idx
			max_Y[i] = Y[curr_max_idx]			
			# max_Y[i] = max(Y[i], maximum(max_Y[1:i-1]))
			if use_Y2
				max_Y2[i] = Y2[curr_max_idx]
			end
		end
	end
	if use_Y2
		return max_Y, max_Y2
	else
		return max_Y
	end
end

# ╔═╡ b4689f1e-e0fd-4bdf-b257-025357fbdcfe
max_returns_pm, max_returns_ho = rolling_max(pm_μ, ho_μ[1:end-1]); # HO has extra run

# ╔═╡ c0da29e4-fcef-4d3a-8552-e0ebe1664947
# max_returns_ho = rolling_max(ho_μ);

# ╔═╡ 61374576-9bda-457b-9cad-bad116444c93
# begin
# 	plot(title="mean returns", ylabel="returns", xlabel="iteration")
# 	hline!([lavi_results[1]], ribbon=lavi_results[2], fillalpha=0.2, ls=:dash, lw=1, c=:black, label="LAVI")

# 	plot!(max_returns_pm, lw=2, label="BetaZero (data collection)", ls=:dash, c=:crimson, alpha=0.5)

# 	plot!(max_returns_ho, lw=2, label="BetaZero (holdout)", c=:darkgreen)
# end

# ╔═╡ abb43a1b-2099-4685-81b8-ccb660c3b91e
begin
	plot(title="mean returns", ylabel="returns", xlabel="iteration")
	hline!([lavi_results[1]], ribbon=lavi_results[2], fillalpha=0.2, ls=:dash, lw=1, c=:black, label="LAVI")

	plot!(pm_μ, ribbon=pm_stderr, fillalpha=0.2, lw=2, label="BetaZero (data collection)", ls=:dash, c=:crimson, alpha=0.5)

	plot!(ho_μ, ribbon=ho_stderr, fillalpha=0.2, lw=2, label="BetaZero (holdout)", c=:darkgreen)
end

# ╔═╡ 1f0c9888-b894-4ef1-9bff-f594f97858dd
ho_acc_μ[23], ho_acc_σ[23]

# ╔═╡ cfc6721e-b68f-4c2f-8d5a-0e5553f7e049
max_accs_pm, max_accs_ho = rolling_max(pm_acc_μ, ho_acc_μ[1:end-1]);# HO has extra run

# ╔═╡ f1a83226-62cf-4942-8d9c-7eb957021f8f
max_accs_pm[end]

# ╔═╡ cbaae95d-583a-4acc-bbb5-0f9c527ad9f8
max_accs_ho[end]

# ╔═╡ 520d50a4-8d67-448c-b8b1-4abdec40b369
# begin
# 	plot(title="accuracy", ylabel="accuracy", xlabel="iteration")
# 	hline!([lavi_acc_results[1]], ribbon=lavi_acc_results[2], fillalpha=0.2, ls=:dash, lw=1, c=:black, label="LAVI")

# 	plot!(max_accs_pm, lw=2, label="BetaZero (data collection)", ls=:dash, c=:crimson, alpha=0.5)

# 	plot!(max_accs_ho, lw=2, label="BetaZero (holdout)", c=:darkgreen)

# 	ylims!(ylims()[1], 1.01)
# end

# ╔═╡ e60856cf-9e2e-4759-806e-409746af824b
begin
	plot(title="accuracy", ylabel="accuracy", xlabel="iteration")
	hline!([lavi_acc_results[1]], ribbon=lavi_acc_results[2], fillalpha=0.2, ls=:dash, lw=1, c=:black, label="LAVI")

	plot!(pm_acc_μ, ribbon=pm_acc_stderr, fillalpha=0.2, lw=2, label="BetaZero (data collection)", ls=:dash, c=:crimson, alpha=0.5)

	plot!(ho_acc_μ, ribbon=ho_acc_stderr, fillalpha=0.2, lw=2, label="BetaZero (holdout)", c=:darkgreen)

	ylims!(ylims()[1], 1.01)
end

# ╔═╡ a88be53e-b4df-4146-a73e-81259c6214d5
md"""
---
"""

# ╔═╡ 1c70d431-0549-4736-8131-287ad37f3c1e
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
BetaZero = "c2a2f090-4363-4bef-a985-c7cb903c4681"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
GaussianProcesses = "891a1506-143c-57d2-908e-e1f8e92e6de9"
GridInterpolations = "bb4c363b-b914-514b-8517-4eb369bc008a"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LightDark = "a1c7e911-b2d9-4cdc-ae72-88f9232b4333"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LocalApproximationValueIteration = "a40420fb-f401-52da-a663-f502e5b95060"
LocalFunctionApproximation = "db97f5ab-fc25-52dd-a8f9-02a257c35074"
MCTS = "e12ccd36-dcad-5f33-8774-9175229e7b33"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
POMDPModels = "355abbd5-f08e-5560-ac9e-8b5f2592a0ca"
POMDPTools = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
ParticleFilters = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BSON = "~0.3.6"
BetaZero = "~0.1.0"
DataStructures = "~0.18.13"
Distributions = "~0.25.81"
Flux = "~0.13.13"
GaussianProcesses = "~0.12.5"
GridInterpolations = "~1.1.2"
Images = "~0.25.2"
Interpolations = "~0.14.7"
LightDark = "~0.1.0"
LocalApproximationValueIteration = "~0.4.3"
LocalFunctionApproximation = "~1.1.0"
MCTS = "~0.5.1"
NNlib = "~0.8.19"
Optim = "~1.7.4"
POMDPModels = "~0.4.16"
POMDPTools = "~0.1.3"
POMDPs = "~0.9.5"
Parameters = "~0.12.3"
ParticleFilters = "~0.5.4"
Plots = "~1.38.5"
PlutoUI = "~0.7.50"
Revise = "~3.5.1"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.3"
manifest_format = "2.0"
project_hash = "6cf48ad48da3b094b23a00d48a78d78902a689be"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "StaticArrays", "Test"]
git-tree-sha1 = "beabc31fa319f9de4d16372bff31b4801e43d32c"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.28"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0310e08cb19f5da31d08341c6120c047598f5b9c"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.5.0"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4d9946e51e24f5e509779e3e2c06281a733914c2"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BSON]]
git-tree-sha1 = "86e9781ac28f4e80e9b98f7f96eae21891332ac2"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BetaZero]]
deps = ["BSON", "DataStructures", "Distributed", "Distributions", "Flux", "GaussianProcesses", "LinearAlgebra", "MCTS", "Optim", "POMDPTools", "POMDPs", "Parameters", "Plots", "ProgressMeter", "Random", "Statistics", "StatsBase", "Suppressor", "UnicodePlots"]
path = "../../home/mossr/src/scerf/BetaZero"
uuid = "c2a2f090-4363-4bef-a985-c7cb903c4681"
version = "0.1.0"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "edff14c60784c8f7191a62a23b15a421185bc8a8"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.0.1"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "75d7896d1ec079ef10d3aee8f3668c11354c03a1"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.2.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "58dd8ec29f54f08c04b052d2c2fa6760b4f4b3a4"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.1.1"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "d3e6ccd30f84936c1a3a53d622d85d7d3f9b9486"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.2.3+2"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "57011df4fce448828165e566af9befa2ea94350a"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.6.0+3"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "7d20c2fb8ab838e41069398685e7b6b5f89ed85b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.48.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "0e5c14c3bb8a61b3d53b2c0620570c332c8d0663"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.2.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "bf6570a34c850f99407b494757f5d7ad233a7257"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.5"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "89a9db8d28102b094992472d333674bd1a83ce2a"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.1"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.D3Trees]]
deps = ["AbstractTrees", "HTTP", "JSON", "Random", "Sockets"]
git-tree-sha1 = "cace6d05f71aeefe7ffd6f955a0725271f2b6cd5"
uuid = "e3df1716-f71e-5df9-9e2d-98e193103c45"
version = "0.3.3"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics"]
git-tree-sha1 = "a5b88815e6984e9f3256b6ca0dc63109b16a506f"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.9.2"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "9258430c176319dc882efa4088e2ff882a0cb1f1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.81"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "e1c40d78de68e9a2be565f0202693a158ec9ad85"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.11"

[[deps.ElasticPDMats]]
deps = ["LinearAlgebra", "MacroTools", "PDMats"]
git-tree-sha1 = "5157c93fe9431a041e4cd84265dfce3d53a52323"
uuid = "2904ab23-551e-5aed-883f-487f97af5226"
version = "0.2.2"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "58d83dd5a78a36205bdfddb82b1bb67682e64487"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.9"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "ed1b56934a2f7a65035976985da71b6a65b4f2cf"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.18.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Zygote"]
git-tree-sha1 = "4ff3a1d7b0dd38f2fc38e813bc801f817639c1f2"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.13"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "a69dd6db8a809f78846ff259298678f0d6212180"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.34"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7ed0833a55979d3d2658a60b901469748a6b9a7c"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "a28f752ffab0ccd6660fc7af5ad1c9ad176f45f7"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.6.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "95185985a5d2388c6d0fedb06181ad4ddd40e0cb"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.17.2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "660b2ea2ec2b010bb02823c6d0ff6afd9bdc5c16"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d5e1fd17ac7f3aa4c5287a61ee28d4f8b8e98873"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.7+0"

[[deps.GaussianProcesses]]
deps = ["Distances", "Distributions", "ElasticArrays", "ElasticPDMats", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra", "Optim", "PDMats", "Printf", "ProgressMeter", "Random", "RecipesBase", "ScikitLearnBase", "SpecialFunctions", "StaticArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "31749ff6868caf6dd50902eec652a724071dbed3"
uuid = "891a1506-143c-57d2-908e-e1f8e92e6de9"
version = "0.12.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1cf1d7dcb4bc32d7b4a5add4232db3750c27ecb4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.8.0"

[[deps.GridInterpolations]]
deps = ["LinearAlgebra", "Printf", "StaticArrays"]
git-tree-sha1 = "9f82426be865173d1488fa4ae73999f59a17deaf"
uuid = "bb4c363b-b914-514b-8517-4eb369bc008a"
version = "1.1.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "2af2fe19f0d5799311a6491267a14817ad9fbd20"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.8"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b1798a4a6b9aafb530f8f0c4a7b2eb5501e2f2a3"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.16"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SnoopPrecompile", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "f265e53558fbbf23e0d54e4fab7106c0f2a9e576"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.3"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "124626988534986113cfd876e3093e4a03890f58"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "e7c68ab3df4a75511ba33fc5d8d9098007b579a8"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.2"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "5985d467623f106523ed8351f255642b5141e7be"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.4"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "fb0b597b4928e29fed0597724cfbb5940974f8ca"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.0"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "ce28c68c900eed3cdbfa418be66ed053e54d4f56"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.7"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "03d1301b7ec885b266c0f816f338368c6c0b81bd"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "c3244ef42b7d4508c638339df1bdbf4353e144db"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.30"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "106b6aa272f294ba47e96bd3acbabdc0407b5c60"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "d9ae7a9081d9b1a3b2a5c1d3dac5e2fdaafbd538"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.22"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "df115c31f5c163697eede495918d8e85045c8f04"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.16.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "7718cf44439c676bc0ec66a87099f41015a522d6"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+2"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightDark]]
deps = ["Distributions", "POMDPModels", "POMDPTools", "POMDPs", "Parameters", "ParticleFilters", "Random"]
path = "../../home/mossr/src/scerf/BetaZero/models/LightDark"
uuid = "a1c7e911-b2d9-4cdc-ae72-88f9232b4333"
version = "0.1.0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LocalApproximationValueIteration]]
deps = ["LocalFunctionApproximation", "POMDPLinter", "POMDPTools", "POMDPs", "Printf", "Random"]
git-tree-sha1 = "30df0ee83bb17c6c541b18375481d7ce2b17f909"
uuid = "a40420fb-f401-52da-a663-f502e5b95060"
version = "0.4.3"

[[deps.LocalFunctionApproximation]]
deps = ["Distances", "GridInterpolations", "NearestNeighbors"]
git-tree-sha1 = "0a6fc6de62ca396ed528a816ad26cdba640221e5"
uuid = "db97f5ab-fc25-52dd-a8f9-02a257c35074"
version = "1.1.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MCTS]]
deps = ["Colors", "D3Trees", "POMDPLinter", "POMDPTools", "POMDPs", "Printf", "ProgressMeter", "Random"]
git-tree-sha1 = "48f7a1f54843f18a98b6dc6cd2edba9db70bdcb8"
uuid = "e12ccd36-dcad-5f33-8774-9175229e7b33"
version = "0.5.1"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "FoldsThreads", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "f69cdbb5b7c630c02481d81d50eac43697084fe0"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.1"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.MarchingCubes]]
deps = ["SnoopPrecompile", "StaticArrays"]
git-tree-sha1 = "55aaf3fdf414b691a15875cfe5edb6e0daf4625a"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.6"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "4d5917a26ca33c66c8e5ca3247bd163624d35493"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "33ad5a19dc6730d592d8ce91c14354d758e53b0e"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.19"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics", "cuDNN"]
git-tree-sha1 = "f94a9684394ff0d325cc12b06da7032d8be01aaf"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "f511fca956ed9e70b80cd3417bb8c2dde4b68644"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.3"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "1903afc76b7d01719d9c30d3c7d501b61db96721"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.4"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "e5a1825d3d53aa4ad4fb42bd4927011ad4a78c3d"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.15"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "95a4038d1011dfdbde7cecd2ad0ac411e53ab1bc"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.10.1"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[deps.POMDPModels]]
deps = ["ColorSchemes", "Compose", "Distributions", "POMDPTools", "POMDPs", "Parameters", "Printf", "Random", "StaticArrays", "StatsBase"]
git-tree-sha1 = "289a869e7a4816fc353e8a292328056a497e4efd"
uuid = "355abbd5-f08e-5560-ac9e-8b5f2592a0ca"
version = "0.4.16"

[[deps.POMDPTools]]
deps = ["CommonRLInterface", "DataFrames", "Distributed", "Distributions", "LinearAlgebra", "NamedTupleTools", "POMDPLinter", "POMDPs", "Parameters", "ProgressMeter", "Random", "Reexport", "SparseArrays", "Statistics", "StatsBase", "Tricks", "UnicodePlots"]
git-tree-sha1 = "a8e1f4af77844c93f4aa9511f5cf2b019db8f4e3"
uuid = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
version = "0.1.3"

[[deps.POMDPs]]
deps = ["Distributions", "Graphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "9ab2df9294d0b23def1e5274a0ebf691adc8f782"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.5"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6f4fbcd1ad45905a5dee3f4256fabb49aa2110c6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.7"

[[deps.ParticleFilters]]
deps = ["LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "1e993c3f16caaf3296fdbb8a4626cc28527d8867"
uuid = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
version = "0.5.4"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "8ac949bd0ebc46a44afb1fdca1094554a84b086e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "7a1a306b72cfa60634f03a911405f4e64d1b718b"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "90cb983381a9dc7d3dff5fb2d1ee52cd59877412"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "9480500060044fd25a1c341da53f34df7443c2f2"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.4"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "7d0b07df35fccf9b866a94bcab98822a87a3cb6f"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.3.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2d7d9e1ddadc8407ffd460e24218e37ef52dd9a3"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.16"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "b03a3b745aa49b566f128977a7dd1be8711c5e71"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.14"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.Suppressor]]
git-tree-sha1 = "c6ed566db2fe3931292865b966d6d140b7ef32a9"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "7e6b0e3e571be0b4dd4d2a9a3a83b65c04351ccc"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.3"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f2fd3f288dfc6f507b0c3a2eb3bac009251e548b"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.22"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c42fa452a60f022e9e087823b47e5a5f8adc53d5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.75"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "Requires", "SnoopPrecompile", "SparseArrays", "StaticArrays", "StatsBase"]
git-tree-sha1 = "ef00b38d086414a54d679d81ced90fb7b0f03909"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.4.0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "e1af683167eea952684188f5e1e29b9cabc2e5f9"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.55"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDNN_jll"]
git-tree-sha1 = "c0ffcb38d1e8c0bbcd3dab2559cf9c369130b2f2"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.0.1"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─40b78b0d-ddd0-4880-92ba-6bb6c5cb1668
# ╠═2a972d56-b232-11ed-0f98-b16501aaff21
# ╠═5a5c8125-d8ed-4664-b5a2-a20819a81920
# ╠═75017254-95fe-4958-a3d2-1ea0b9b8f0a2
# ╠═e38aac74-92a7-4ae7-97c9-57ec6dc9afcd
# ╟─7e65e65a-6cf0-4e3b-afc0-961aeddde9f9
# ╠═48ad6bac-68da-4742-a789-2e5e5b8eed10
# ╠═0c73c232-95a7-4883-811f-614e676e142a
# ╠═83e53386-1b8f-4308-863a-fe6fcfa933b6
# ╠═c0f29e13-bdce-4908-9e9a-6001c88e5ed9
# ╠═1bb8df23-575a-479b-a4ce-3ceb3eb7acbf
# ╠═8c7e2291-b846-4659-a3bd-f6abacdd3130
# ╠═6309e3f0-29c9-4963-a2d1-f6fcb5f3ada4
# ╠═80c2be9e-a242-4e9d-a082-8c61cfe5ae74
# ╠═3af7e1d4-5664-4e71-a075-5695e649bbfa
# ╠═243f9939-0d54-457d-be7d-6f0648f4fa63
# ╟─db354fe2-6fe4-4593-8c8b-9c6bbf6f05e7
# ╠═fb09d547-377a-4e34-9e5f-87d06a8c12e2
# ╠═41d937dd-058b-469d-bc1f-7b70e26bf0c4
# ╠═87c160e9-63f2-4628-b4aa-187e1b85b46d
# ╠═51afcc23-64d6-4da2-91aa-8fe81d01440b
# ╠═4408f005-e0bb-4f96-9f7e-38b1d61c7c9b
# ╟─90c4edd0-0215-47df-ba5b-c645576a69f2
# ╠═07aa5bc1-0a89-4de5-aba7-b74265b7bbed
# ╠═03ff8055-8725-4935-8a09-cb1ea6bf6e35
# ╠═42de9867-be47-44f4-80ec-15a066a45b8c
# ╠═1208652f-3ab9-4379-b0ba-b80f29977b3c
# ╠═ac5456ea-4211-480f-90c8-b50b4a28d0b9
# ╠═769e3159-bf5e-4294-b7b9-9e49a32406a1
# ╠═42b011d7-268b-4c9f-94d9-bbc4981847c1
# ╠═8d465a2a-2138-4415-92aa-f0c4c20bdd6d
# ╠═eb6537de-b4d2-49bd-9a20-529240fa4e98
# ╠═f105a961-d94e-4c3d-b1d2-6dd077922dc7
# ╠═812bd7a7-70af-4581-9170-8ce8af05bd14
# ╠═ae9322da-f369-4f36-9e92-861a1b0f2c4c
# ╠═ebf5d5bd-6f0e-45ed-af7b-fa06c3140128
# ╠═c84e0d55-33e3-415e-b108-51efd3a13717
# ╠═848d0d9c-f036-44a1-8ce4-ba12798cd538
# ╠═9b4afdca-9c87-4e88-8199-7fa17c2403fe
# ╠═029202f4-ba86-457a-911e-30f02ef99f47
# ╠═547c9e31-6372-4625-b1a7-212adb2e5438
# ╟─c6c3bcb9-ecd0-494b-9fa5-11c35c722396
# ╠═3bbc44ce-b0b2-4c81-8ec9-d4819bd8ba49
# ╠═bd73ffa0-3a73-4922-b7a8-3ffc62f8da25
# ╠═ee234170-f86b-4b03-9256-7cbdc644d74a
# ╠═1f52fb7c-56e2-4fc4-9c00-b35b523ac51f
# ╠═56e2593e-cdf0-4cde-830c-09af3daa7e17
# ╠═8689d008-0a28-45c1-b35c-0f17c990cb5c
# ╟─30e52cf5-37d4-4b85-908c-43990d6592bd
# ╠═be289f06-ea10-43d8-9c03-0a14aa1579a4
# ╠═5565d9e9-bb35-4bb0-91ad-73662e9d6136
# ╠═4e8af1b8-7b5f-442d-b28b-0b8d7bdf1a7d
# ╠═ec606b95-cb6c-4a54-8e1d-d1a1881457d9
# ╠═d75b9ea2-6a15-4e2d-ab78-0eb16ea54b11
# ╠═c59a36ee-11e6-420c-9c27-16cbe23cd6e6
# ╠═c6aa050f-646e-4033-98ef-426c774b2bc9
# ╠═157e9d3c-3a4d-4426-86f3-1d731a8da122
# ╠═fd432cf8-1e7e-475b-9c41-46d18ffe2335
# ╠═3b2cffc2-1f52-453e-a244-a3603c829564
# ╠═007c03ca-8a6e-414f-89bd-7fa29fdbf540
# ╠═1af2cbfe-4ccf-4b4e-a253-ab6b63d037f1
# ╠═3aae1c42-4092-425f-8086-cbe0518a2692
# ╠═609d9262-10b9-4e9e-9bda-5ad50338ac62
# ╠═8ab1e3f5-710c-4ecf-9cc8-c1e2d914a011
# ╠═ac56db3d-ea72-459e-a072-920edd03cb67
# ╠═40b599eb-e068-4faf-b54e-6dca4896d4a2
# ╠═4c81fcc5-b50b-4ecd-a0f4-f3ddfe3fad19
# ╠═1515aa19-f8b5-4d26-93a8-3afd3e144ad1
# ╠═70b0c991-ca46-4434-ac21-4f7785755cb4
# ╟─828477f3-d19b-49c2-9827-2cc2d86233e6
# ╟─47fb1fc3-dbeb-4696-ae7a-d0f049c3ccf3
# ╠═598bf115-667e-49e5-bdad-bcef4e3e1cbc
# ╠═7ef35aa9-8558-40aa-abee-1f3b671004c0
# ╠═35c24d73-3c31-4d8b-9ab3-3acc359911d7
# ╠═bfa256df-3c81-4d55-8384-aa277ea3a645
# ╠═98b53fd7-d853-4869-849b-e6875ab89c5d
# ╠═63364798-48c4-4a7b-8f91-29bda69307f1
# ╠═9c523eb3-94ce-4fe6-a241-b9fa7d1df25b
# ╠═05d7ecc1-34c4-4b34-bd3b-4cb3ad0267b7
# ╠═1e5d06fe-7036-4a13-83ec-7cf301573c25
# ╠═f60f4c7a-9b4b-48c2-921b-0fa4214ba435
# ╠═bc965027-5da1-4377-9ed5-e72b9452d526
# ╠═54b15814-9d64-4799-bdfb-514e2a3e48bb
# ╠═bb851367-0421-4e36-b20b-d11fc57874b5
# ╠═e6732dc1-ab93-4d61-b215-168cbb00c0ee
# ╠═ff8d9df0-7635-4f9e-a3ee-c279977f8173
# ╠═86f1ed3f-d58a-44bf-ba6c-dbb2b04b4e8c
# ╠═64f10375-ad0b-4640-b117-fe96bf1ea9e1
# ╠═0dea654f-e4de-4ecf-8ad4-ca2ec76136c9
# ╠═2bd13b69-31fc-4885-a5f5-d95457873445
# ╠═8c6f978c-afe6-481e-a52d-63fb597737d5
# ╠═b0caab16-2b08-4683-b14f-ba3bcf3275ef
# ╠═e84ca904-a339-4d43-a3ae-41988868ecee
# ╠═f278d28d-ccd0-411a-a74e-fbaa4efca965
# ╠═df1ca98b-68a2-413d-9253-fe709d463ce9
# ╠═f9f24938-4e9e-46bf-926a-646c937e48e2
# ╟─c5591821-7615-4351-9a01-12c82efb746f
# ╠═8b172603-92f0-4750-823e-02fc61eeaa6b
# ╠═1d309991-1c38-40e3-874a-884dc83709da
# ╠═2a06238b-06ca-453d-a797-162f183cd869
# ╟─5658517d-5b0e-4a58-982d-e7cb08b3522e
# ╠═38e1c560-beaf-4c2e-8f5c-d6b08a4d3273
# ╠═cd4dab5b-a7cd-4e07-b6aa-98d1220ad0c8
# ╠═0feeb06c-3f6d-4ec7-b795-9a1c52fc0e69
# ╟─11d10167-0a74-4153-b618-52ae07889324
# ╠═4f5e76f2-97d3-418e-a1e5-2fc558fbddcb
# ╠═7f8ae7e3-8144-4d38-b398-edecf5fc396f
# ╠═607ed6b8-af59-402a-b9ba-6786bb3f248c
# ╟─1bea8485-6ccd-4108-a97d-9669f6669da8
# ╠═14ea3b14-a718-43de-a126-7910fb4e02c1
# ╠═2f598533-6de2-4029-b3f9-7a2904c8d766
# ╠═f98dd3af-2af8-40b1-b71e-d8e842b411c3
# ╠═74f2b6ec-4e5a-4887-8e31-3cd41d54b375
# ╠═ccf70808-5103-4279-846d-237f888fba01
# ╠═6d822d41-f49a-4d11-8266-e41a02f5aaa1
# ╠═2acb5d21-e5b4-446c-80d0-ee98f3bbd4a7
# ╠═dd9e1fcc-79c1-4452-b032-51f918303f17
# ╠═e20fbafd-0f76-4862-a1b3-a7fff41b47b1
# ╠═ad45a730-5950-43b6-ba6b-82ff9373c59d
# ╠═73e8ca7f-cf0a-431f-8625-7e317224bed4
# ╟─6f7bc73f-3961-45af-90ff-002e02dbe9eb
# ╠═9186d51c-dac7-4b21-b4eb-3cd724bb1a3e
# ╠═ce006bd6-d691-4efb-b740-4ddc258416c4
# ╠═f97f9786-5cc7-478f-997c-2fb26053a504
# ╠═d1296cde-fe14-4ca0-84bf-540c7752125e
# ╠═a2f127dc-d372-40ec-851b-06022890710f
# ╠═451a9c77-f76e-4912-bb3e-5c02a24b1b84
# ╠═1709523a-8674-43a7-94c0-de9fc9a5479a
# ╠═898c7dfe-fc02-4810-8d37-4b06247feea9
# ╠═d2955613-e976-4b5b-81ba-1c9454ec2c03
# ╠═fdccfa1f-da99-4b59-b647-a426ecd7566a
# ╠═652b3290-815a-4e33-80c3-3d7ef84bd8af
# ╠═5853347f-86e7-459d-83cd-ba23239ffb98
# ╠═4b0b6f23-1fca-4907-bdb3-eafb3bbc9606
# ╠═6f2b96cc-c0cf-42fa-8cec-380ff48d9106
# ╠═343bdb4e-f9ba-4c10-a711-750d695652f7
# ╠═6213273f-0417-4cbc-9eff-9add06948401
# ╠═8d422b96-b8ef-4d36-ab80-41d404cf4ff0
# ╟─6572e23c-d4f4-4803-8034-6613fdf18960
# ╠═bda2e95f-b22f-4ba1-95ba-f973669e31d9
# ╠═47057cbd-508c-48d8-a2aa-59f3a5296481
# ╠═2a860403-efcf-40d9-81a0-bd0f2b76081a
# ╠═ac665280-9b69-4eb4-b4ca-e794fb3da01b
# ╠═c455f35d-25d7-44d7-90dd-a2cb9c42020c
# ╠═b4a20910-2b1c-4697-9140-eb619fffbe40
# ╠═f2e9c419-d2ca-4bc8-8f94-2115f192decf
# ╠═8dca08c1-e469-44a1-95ad-af4c76ba2729
# ╠═b2c7630b-3673-4158-9e91-4fbd2f471d08
# ╠═92d0d38f-9fb6-4c08-b569-6b336d8fba8a
# ╠═32c79abb-122c-421a-831e-28b4faaad279
# ╠═c6a3bcf0-c7f4-468a-9884-f7fe4231ab5d
# ╠═ce515ed9-bac3-4b48-977f-c94937eebe8f
# ╠═42999b4d-7b41-41b8-b719-2e375d890c97
# ╠═2694d4ba-5eaf-4bf7-a9eb-a9e407163cf1
# ╠═7b237f3d-6898-4c57-b672-7449f8a485b0
# ╠═bcd84550-5656-4997-a4ce-1167852c7eec
# ╠═f7a13b63-f7f5-41cf-bad4-7d8a5f588c6d
# ╠═8f196424-1b11-4ee5-bc58-ee5284fdf098
# ╠═d9acdd13-cb52-4ea3-a493-2534f43e236a
# ╠═894ee284-a417-4017-9696-6f741706f179
# ╠═c32085e0-8f24-4e87-bedc-d18e8197e575
# ╠═dc49a10c-4648-4e8d-bc85-cc6cfc073c5c
# ╠═7d7670e3-d07b-4175-b399-68327a1c3b03
# ╠═886e5e3c-ec60-4da3-bf49-18e811578853
# ╠═59ca30e3-7397-4245-8985-c25795e46750
# ╠═8d818d7f-0f98-4131-a723-12fd164d218f
# ╠═31684f7c-f5ad-4ea9-993e-cd19c38a8eae
# ╠═fa79f064-81df-476c-a3fb-5271bee3cd5a
# ╠═410bdb7a-a201-47fb-9a92-9910b5b93ca0
# ╠═8f0506e3-30df-4093-afa1-cb035673cf01
# ╠═c5332871-a0d3-4586-8be3-643fcd9bb392
# ╠═00a84f88-9d7e-4344-b813-f66b6a70c394
# ╠═f067942f-86a6-4edc-bcf0-78fecb6455a0
# ╟─0cf1d477-d750-4a45-8bd7-7a3c061c6532
# ╠═5be544ba-d05f-4e4e-b88e-3f7cc1ee79c4
# ╠═6b2514dd-4312-4030-b4b0-d26d6f936c5d
# ╠═46166104-92e8-4f5a-bcd4-35435f18aa63
# ╠═b3e3c159-31f8-4d3f-b062-43dc4b940db6
# ╠═12458ba9-10ed-4ebc-88fd-140f86b79920
# ╠═a8782a5e-9a51-470b-9d89-c8188c04dc12
# ╠═31a99d07-aa61-46e1-827f-05d3023d0a09
# ╠═93be835a-a3e0-4147-a2ad-3a86d93a34b5
# ╠═980c2630-b66b-423f-8153-3a979c16bdf6
# ╠═0255a5cc-7a77-42dd-b86e-b9e1d2c88079
# ╠═61284855-5286-487a-b6b0-f8a24adb28a3
# ╟─d4d9e5b3-3f6c-4936-9592-8b08ed02a09e
# ╟─e56a6370-00e5-4305-8062-8106dc2df9cb
# ╠═88362011-8e3b-45de-b451-57265459aa71
# ╠═87d7505d-2edb-401d-ac2e-9386f5ccad5f
# ╠═baa3ab4f-be38-4851-a851-4a37b38a3f66
# ╠═fc28e360-a595-46e3-ba32-d917ce0b9f63
# ╠═2e512cda-bba3-4cc9-9110-132ffa37e9e5
# ╠═1ae1a1e8-aaec-47b2-9ced-6fd8afc48b9e
# ╠═01f7dcdf-fff2-4b58-93da-585c1cebd193
# ╠═b9cb080f-087f-4d4b-b575-2190652b92ea
# ╠═8f2b6e82-4e3d-440d-ba16-ad043b833b97
# ╠═9ee1c3b0-d600-402f-bc42-4fa2feeee807
# ╠═c361ac8e-51bf-436b-a195-b728ceceb4ce
# ╠═eebb4757-83de-490a-84d5-52f988708c3a
# ╠═ddbf075e-539e-49a7-8a1b-a8e7b6eb644f
# ╠═1fa63234-37e4-4b12-a064-6a55b2ee0b67
# ╟─047c2c69-d662-4dba-b8b6-c25167178ecb
# ╠═0f8578cd-37f5-4ecd-b9eb-6a6351750fa9
# ╠═743cf6ad-ddaa-403f-8ab8-83513c804134
# ╠═7c7b0c5a-ca81-4906-b0a2-ab0c2f1b7782
# ╠═351d5b4f-4347-441a-88ef-06d54eaf0c16
# ╠═058622d9-db81-4a0b-a7a7-dcc5fb765090
# ╠═e52988c7-a2d1-411c-a0cf-119180b2d0e0
# ╠═fd9ece7a-2466-47b8-94b5-02dd5f3181ca
# ╠═d1977228-b7a2-4a87-b556-c111039509ce
# ╠═ad0b9cfa-a1e6-479c-99ae-179b778c716b
# ╠═daafbdec-0311-4981-a8e2-201edeca5b1e
# ╠═a68e35de-883c-46ec-ab66-6a77b43a04c7
# ╠═99608de8-ef1c-473c-a8d4-f21d9800e389
# ╠═b4689f1e-e0fd-4bdf-b257-025357fbdcfe
# ╠═c0da29e4-fcef-4d3a-8552-e0ebe1664947
# ╠═61374576-9bda-457b-9cad-bad116444c93
# ╠═abb43a1b-2099-4685-81b8-ccb660c3b91e
# ╠═1f0c9888-b894-4ef1-9bff-f594f97858dd
# ╠═cfc6721e-b68f-4c2f-8d5a-0e5553f7e049
# ╠═f1a83226-62cf-4942-8d9c-7eb957021f8f
# ╠═cbaae95d-583a-4acc-bbb5-0f9c527ad9f8
# ╠═520d50a4-8d67-448c-b8b1-4abdec40b369
# ╠═e60856cf-9e2e-4759-806e-409746af824b
# ╟─a88be53e-b4df-4146-a73e-81259c6214d5
# ╠═1c70d431-0549-4736-8131-287ad37f3c1e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
