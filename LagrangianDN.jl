using Lux, LuxCore, ForwardDiff, ReverseDiff, Zygote, Enzyme

struct LagrangianNN{M <: LuxCore.AbstractExplicitLayer} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    ad
end

function LagrangianNN(model; ad = AutoForwardDiff())
    @assert ad isa AutoForwardDiff || ad isa AutoZygote || ad isa AutoEnzyme
    !(model isa Lux.AbstractExplicitLayer) && (model = Lux.transform(model))
    return LagrangianNN(model, ad)
end

function (lnn::LagrangianNN{ <: LuxCore.AbstractExplicitLayer})(state, p, st)
    model = Lux.StatefulLuxLayer(lnn.model, nothing, st)
    return model(state, p)[1], model.st
end


using LinearAlgebra

function EulerLagrange(lnn::LagrangianNN{ <: LuxCore.AbstractExplicitLayer}, st)
    function (dstate, state, p, t)
        q = state[1, :]
        qdot = state[2, :]
        
        dLdq = ForwardDiff.gradient(q -> first(lnn(vcat(q, qdot), p, st)), q)
        d_dLdqdot_dq = ForwardDiff.jacobian(q -> ForwardDiff.gradient(qdot -> first(lnn(vcat(q, qdot), p, st)), qdot), q)      
        lambda = 1.e-3  
        H = ForwardDiff.hessian(qdot -> first(lnn(vcat(q, qdot), p, st)), qdot) + lambda * I
        
        dstate[1, :] = qdot
        dstate[2, :] = H \ (dLdq - d_dLdqdot_dq * qdot)
    end
end


using OrdinaryDiffEq, DiffEqFlux

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
    prob = OrdinaryDiffEq.ODEProblem(EulerLagrange(nlde.model, st), state, nlde.tspan, ps)
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())
    return Array(OrdinaryDiffEq.solve(prob, nlde.args...; sensealg, nlde.kwargs...)), st
end


using Plots

function trueODE(du, u, p, t)
    du[1] = u[2]
    du[2] = - 0.5 * 30 * u[1]
end

u0 = Float32[2.0; -0.3]
tspan = (0.0, 1.0)

datasize = 30
tsteps = range(tspan[1], tspan[2]; length = datasize)

true_ode_prob = ODEProblem(trueODE, u0, tspan)
println("Solving the trueODE...")
true_sol = Array(solve(true_ode_prob, Tsit5(); saveat = tsteps))
println("Done.")

using Random, ComponentArrays, Flux

rng = Random.default_rng()

println("Setting up LNN and NLDE...")
lnn = LagrangianNN(Flux.Chain(Flux.Dense(2 => 10, tanh), Flux.Dense(10 => 1)); ad = AutoForwardDiff())
ps, st = Lux.setup(rng, lnn)
ps = ps |> ComponentArray
model = NeuralLagrangianDE(lnn, tspan, Tsit5(); saveat = tsteps)
println("Done.")

function loss(p)
    pred, _ = model(u0, p, st)
    loss = sum(abs2, true_sol .- pred)
    return loss, pred
end

callback = function (p, l, pred)
    println("loss ", l)
    return false
end

println("Solving the LNN...")
l, pred = loss(ps)
println("Done.")
callback(ps, l, pred)
# println("Plotting...")
# plt = scatter(tsteps, true_sol[1, :]; label = "data")
# scatter!(plt, tsteps, pred[1, :]; label = "prediction")
# display(plt)
# println("Done.")


using Optimization, OptimizationOptimisers

println("Setting up OptProb...")
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((u, p) -> loss(u), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)
println("Done.")

println("Solving the OptProb...")
result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 15)
println("Done.")
l, pred = loss(ps)
callback(ps, l, pred)
