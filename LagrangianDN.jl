using ComponentArrays, LinearAlgebra, DiffEqFlux, OrdinaryDiffEq, ForwardDiff, Zygote, Random

# function EulerLagrange!(lagrangian, dstate, state, p, t)
#     q = state[1, :]
#     qdot = state[2, :]

#     dLdq = ForwardDiff.gradient(q -> lagrangian(hcat(q, qdot), p, t), q)
#     d_dLdqdot_dq = ForwardDiff.jacobian(q -> ForwardDiff.gradient(qdot -> lagrangian(hcat(q, qdot), p, t), qdot), q)
#     H = ForwardDiff.hessian(qdot -> lagrangian(hcat(q, qdot), p, t), qdot)

#     dstate[1, :] = qdot
#     dstate[2, :] = H \ (dLdq - d_dLdqdot_dq * qdot)
# end

# function EulerLagrange_wrapper(lagrangian)
#     return function (dstate, state, p, t)
#         return EulerLagrange!(lagrangian, dstate, state, p, t)
#     end
# end

"""
    LagrangianNN(model; ad = AutoForwardDiff())

Constructs a Lagrangian Neural Network. This neural network is useful for learning
symmetries and conservation laws by supervision on the gradients of the trajectories. It
takes as input a concatenated vector of length `2n` containing the position (of size `n`)
and momentum (of size `n`) of the particles. It then returns the time derivatives for
position and momentum.

!!! note

    This doesn't solve the Lagrangian Problem. Use [`NeuralLagrangianDE`](@ref)
    for such applications.

Arguments:

 1. `model`: A `Flux.Chain` or `Lux.AbstractExplicitLayer` neural network that returns the
    Lagrangian of the system.
 2. `ad`: The autodiff framework to be used for the internal Lagrangian computation. The
    default is `AutoForwardDiff()`.

!!! note

    If training with Zygote, ensure that the `chunksize` for `AutoForwardDiff` is set to
    `nothing`.

"""
struct LagrangianNN{M <: LuxCore.AbstractExplicitLayer} <:
                 Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    ad
end

function LagrangianNN(model; ad = AutoForwardDiff())
    @assert ad isa AutoForwardDiff || ad isa AutoZygote || ad isa AutoEnzyme
    !(model isa Lux.AbstractExplicitLayer) && (model = Lux.transform(model))
    return LagrangianNN(model, ad)
end

function (lnn::LagrangianNN{ <: LuxCore.AbstractExplicitLayer})(state, ps, st)
    model = StatefulLuxLayer(lnn.model, nothing, st)
    return model(state, ps)[1], model.st
end

function EulerLagrange(lnn::LagrangianNN{ <: LuxCore.AbstractExplicitLayer}, st)
    function (dstate, state, ps, t)
        q = state[1, :]
        qdot = state[2, :]
        
        dLdq = ForwardDiff.gradient(q -> first(lnn(vcat(q, qdot), ps, st)), q)
        d_dLdqdot_dq = ForwardDiff.jacobian(q -> ForwardDiff.gradient(qdot -> first(lnn(vcat(q, qdot), ps, st)), qdot), q)        
        H = ForwardDiff.hessian(qdot -> first(lnn(vcat(q, qdot), ps, st)), qdot)
        
        dstate[1, :] = qdot
        dstate[2, :] = pinv(H) * (dLdq - d_dLdqdot_dq * qdot)
    end
end


"""
    NeuralLagrangianDE(model, tspan, args...; kwargs...)

Constructs a Neural Lagrangian DE Layer for solving Lagrangian Problems parameterized by a
Neural Network [`LagrangianNN`](@ref).

Arguments:

  - `model`: A Flux.Chain, Lux.AbstractExplicitLayer, or Lagrangian Neural Network that
    predicts the Lagrangian of the system.
  - `tspan`: The timespan to be solved on.
  - `kwargs`: Additional arguments splatted to the ODE solver. See the
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.
"""
struct NeuralLagrangianDE{M <: LagrangianNN} <: DiffEqFlux.NeuralDELayer
    model::M
    tspan
    args
    kwargs
end

function NeuralLagrangianDE(model, tspan, args...; ad = AutoForwardDiff(), kwargs...)
    hnn = model isa LagrangianNN ? model : LagrangianNN(model; ad)
    return NeuralLagrangianDE(hnn, tspan, args, kwargs)
end

function (nlde::NeuralLagrangianDE)(state, ps, st)
    prob = ODEProblem(EulerLagrange(nlde.model, st), state, nlde.tspan, ps)
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())
    return Array(OrdinaryDiffEq.solve(prob, nlde.args...; sensealg, nlde.kwargs...)), st
end

tspan = (0.0f0, 1.0f0)
tsteps = range(tspan[1], tspan[2]; length = 64)
state = [1.0 2.0 4.0; 3.0 2.0 1.2]
dstate = state

lnn = LagrangianNN(Chain(Dense(6 => 50, tanh), Dense(50 => 1)); ad = AutoForwardDiff())
ps, st = Lux.setup(Random.default_rng(), lnn)
ps = ps |> ComponentArray

# println(first(lnn(state[:], ps, st)))
println(EulerLagrange(lnn, st)(dstate, state, ps, tspan))

model = NeuralLagrangianDE(lnn, tspan, Tsit5(); saveat = tsteps)
xpred = first(model(state, ps, st))[1, 1, :]
println(xpred)

using Plots 
plt = scatter(tsteps, xpred)
display(plt)
# model = NeuralLagrangianDE(lnn, tspan, Tsit5(); saveat = tsteps)

# println(model([1.0; 2.0], ps, st))

# opt = Optimisers.Adam(0.01)
# st_opt = Optimisers.setup(opt, ps)
# loss(data, target, ps) = mean(abs2, first(model(state, ps, st)) .- target)

# initial_loss = loss(data, target, ps)
# println(initial_loss)

# for epoch in 1:100
#     global ps, st_opt
#     gs = last(Zygote.gradient(loss, data, target, ps))
#     st_opt, ps = Optimisers.update!(st_opt, ps, gs)
# end

# final_loss = loss(data, target, ps)

