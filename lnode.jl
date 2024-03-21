using LinearAlgebra, OrdinaryDiffEq, ComponentArrays
using Lux, LuxCore, Flux, DiffEqFlux, ForwardDiff, Zygote, Enzyme
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random, Plots 

struct LagrangianNN{M <: LuxCore.AbstractExplicitLayer} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    ad
end

function LagrangianNN(model; ad = AutoForwardDiff())
    @assert ad isa AutoForwardDiff || ad isa AutoZygote || ad isa AutoEnzyme
    !(model isa Lux.AbstractExplicitLayer) && (model = Lux.transform(model))
    return LagrangianNN(model, ad)
end

function (lnn::LagrangianNN{ <: LuxCore.AbstractExplicitLayer})(u, p, st)
    model = Lux.StatefulLuxLayer(lnn.model, nothing, st)
    q = u[1, :]
    qdot = u[2, :]

    dLdq = ForwardDiff.gradient(q -> first(model(vcat(q, qdot), p)), q)
    d_dLdqdot_dq = ForwardDiff.jacobian(q -> ForwardDiff.gradient(qdot -> first(model(vcat(q, qdot), p)), qdot), q)        
    H = ForwardDiff.hessian(qdot -> first(model(vcat(q, qdot), p)), qdot) + 1e-6 * I
        
    qddot = H \ (dLdq - d_dLdqdot_dq * qdot)
    return vcat(qdot, qddot), model.st
end


rng = Random.default_rng()

lnn = LagrangianNN(Flux.Chain(Flux.Dense(2 => 4, tanh), Flux.Dense(4 => 1)); ad = AutoForwardDiff())
ps, st = Lux.setup(rng, lnn)
ps = ps |> ComponentArray

u0 = Float32[2.0; -0.3]
println(lnn(u0, ps, st))

struct NeuralLagrangianDE{M <: LagrangianNN} <: DiffEqFlux.NeuralDELayer
    model::M
    tspan
    args
    kwargs
end

function NeuralLagrangianDE(model, tspan, args...; ad = AutoForwardDiff(), kwargs...)
    lnn = model isa LagrangianNN ? model : LagrangianNN(model; ad)
    return NeuralLagrangianDE(lnn, tspan, args, kwargs)
end

function (nlde::NeuralLagrangianDE)(state, ps, st)
    model = Lux.StatefulLuxLayer(nlde.model, nothing, st)
    neural_lagrangian(u, p, t) = model(u, p)
    prob = OrdinaryDiffEq.ODEProblem{false}(neural_lagrangian, state, nlde.tspan, ps)
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())
    return Array(DiffEqFlux.solve(prob, nlde.args...; sensealg, nlde.kwargs...)), model.st
end

tspan = (0.0, 1.5)
datasize = 30
tsteps = range(tspan[1], tspan[2]; length = datasize)

function trueODE(du, u, p, t)
    du[1] = u[2]
    du[2] = - 25.0 * u[1]
end

println("Solving the trueODE...")
true_sol = Array(solve(ODEProblem(trueODE, u0, tspan), Tsit5(); saveat = tsteps))
println("Done.")

NeuraL = NeuralLagrangianDE(lnn, tspan, Tsit5(); ad = AutoForwardDiff(), saveat = tsteps)

function loss(target, u0, p)
    pred, _ = NeuraL(u0, p, st)
    loss = sum(abs2, target .- pred)
    return loss, pred
end

println("Making a LNN prediction...")
l, pred = loss(true_sol, u0, ps)
println("loss ", l)

println("Setting up OptProb...")
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((u, p) -> loss(true_sol, u0, u), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)
println("Done.")

println("Solving the OptProb...")
result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.1); maxiters = 500)
# result_neuralode = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = 0.01); allow_f_increases = false)
println("Done.")

fl, pred = loss(true_sol, u0, result_neuralode.u)
println("final loss ", fl)
