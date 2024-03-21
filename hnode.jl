using Lux, Flux
using ForwardDiff, Zygote
using Optimization, OptimizationOptimisers
using DiffEqFlux, OrdinaryDiffEq
using Random, ComponentArrays, Plots, LaTeXStrings


struct HamiltonianNN{M <: Lux.AbstractExplicitLayer} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    ad
end

function HamiltonianNN(model)
    !(model isa Lux.AbstractExplicitLayer) && (model = Lux.transform(model))
    return HamiltonianNN(model, Optimization.AutoZygote())
end

function (hnn::HamiltonianNN{<: Lux.AbstractExplicitLayer})(state, ps, st)
    model = Lux.StatefulLuxLayer(hnn.model, nothing, st)
    dH = first(Zygote.gradient(sum ∘ model, state, ps))
    n = size(state, 1) ÷ 2
    return vcat(selectdim(dH, 1, (n + 1):(2n)), -selectdim(dH, 1, 1:n)), model.st
end


hnn = HamiltonianNN(Flux.Chain(Flux.Dense(2 => 16, tanh), Flux.Dense(16 => 1)))
ps, st = Lux.setup(Random.default_rng(), hnn)
ps = ps |> ComponentArray


π_32 = Float32(π)
tspan = (0.0f0, 1.0f0)
tsteps = range(tspan[1], tspan[2]; length = 64)

q_t = reshape(sin.(2π_32 * tsteps), 1, :)
p_t = reshape(cos.(2π_32 * tsteps), 1, :)
data = vcat(q_t, p_t)

dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t
target = vcat(dqdt, dpdt)

function loss_function(ps, data, target)
    pred, _ = hnn(data, ps, st)
    return sum(abs2, pred .- target), pred
end

opt_func = Optimization.OptimizationFunction((u, p) -> loss_function(u, data, target), Optimization.AutoZygote())
opt_prob = Optimization.OptimizationProblem(opt_func, ps)

function callback(ps, loss, pred; doplot = true)
    println("[Hamiltonian NN] Loss: ", loss)
    if doplot
        plt = scatter(tsteps, target[1, :]; label = "target")
        scatter!(plt, tsteps, pred[1, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end

opt_res = Optimization.solve(opt_prob, OptimizationOptimisers.Adam(0.05f0); 
			     callback, maxiters = 500)
ps_trained = opt_res.u
final_loss, pred = loss_function(ps_trained, data, target)

struct NeuralHamiltonianDE{M <: HamiltonianNN} <: DiffEqFlux.NeuralDELayer
    model::M
    tspan
    args
    kwargs
end

function NeuralHamiltonianDE(model, tspan, args...; kwargs...)
    hnn = model isa HamiltonianNN ? model : HamiltonianNN(model)
    return NeuralHamiltonianDE(hnn, tspan, args, kwargs)
end

function (nhde::NeuralHamiltonianDE)(x, ps, st)
    model = Lux.StatefulLuxLayer(nhde.model, nothing, st)
    neural_hamiltonian(u, p, t) = model(u, p)
    prob = ODEProblem{false}(neural_hamiltonian, x, nhde.tspan, ps)
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())
    return solve(prob, nhde.args...; sensealg, nhde.kwargs...), model.st
end


model = NeuralHamiltonianDE(hnn, tspan, OrdinaryDiffEq.Tsit5(); 
            save_everystep = false, save_start = true, saveat = tsteps)

pred = Array(first(model(data[:, 1], ps_trained, st)))