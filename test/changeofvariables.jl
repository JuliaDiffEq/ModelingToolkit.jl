using ModelingToolkit, OrdinaryDiffEq
using Test, LinearAlgebra


# Change of variables: z = log(x)
# (this implies that x = exp(z) is automatically non-negative)

@parameters t α
@variables x(t)
D = Differential(t)
eqs = [D(x) ~ α*x]

tspan = (0., 1.)
u0 = [x => 1.0]
p = [α => -0.5]

sys = ODESystem(eqs; defaults=u0)
prob = ODEProblem(sys, [], tspan, p)
sol = solve(prob, Tsit5())

@variables z(t)
forward_subs  = [log(x) => z]
backward_subs = [x => exp(z)]
new_sys = changeofvariables(sys, forward_subs, backward_subs)
@test equations(new_sys)[1] == (D(z) ~ α)

new_prob = ODEProblem(new_sys, [], tspan, p)
new_sol = solve(new_prob, Tsit5())

@test isapprox(new_sol[x][end], sol[x][end], atol=1e-4)



# Riccati equation
@parameters t α
@variables x(t)
D = Differential(t)
eqs = [D(x) ~ t^2 + α - x^2]
sys = ODESystem(eqs, defaults=[x=>1.])

@variables z(t)
forward_subs  = [t + α/(x+t) =>    z       ]
backward_subs = [       x    => α/(z-t) - t]

new_sys = changeofvariables(sys, forward_subs, backward_subs; simplify=true, t0=0.)
# output should be equivalent to
# t^2 + α - z^2 + 2   (but this simplification is not found automatically)

tspan = (0., 1.)
p = [α => 1.]
prob = ODEProblem(sys,[],tspan,p)
new_prob = ODEProblem(new_sys,[],tspan,p)

sol = solve(prob, Tsit5())
new_sol = solve(new_prob, Tsit5())

@test isapprox(sol[x][end], new_sol[x][end], rtol=1e-4)


# Linear transformation to diagonal system
@parameters t
@variables x[1:3](t)
D = Differential(t)
A = [0. -1. 0.; -0.5 0.5 0.; 0. 0. -1.]
eqs = D.(x) .~ A*x

tspan = (0., 10.)
u0 = x .=> [1.0, 2.0, -1.0]

sys = ODESystem(eqs; defaults=u0)
prob = ODEProblem(sys,[],tspan)
sol = solve(prob, Tsit5())

T = eigen(A).vectors

@variables z[1:3](t)
forward_subs  = T \ x .=> z
backward_subs = x     .=> T*z

new_sys = changeofvariables(sys, forward_subs, backward_subs; simplify=true)

new_prob = ODEProblem(new_sys, [], tspan, p)
new_sol = solve(new_prob, Tsit5())

# test RHS
new_rhs = [eq.rhs for eq in equations(new_sys)]
new_A = Symbolics.value.(Symbolics.jacobian(new_rhs, z))
@test isapprox(diagm(eigen(A).values), new_A, rtol = 1e-10)
@test isapprox( new_sol[x[1],end], sol[x[1],end], rtol=1e-4)
