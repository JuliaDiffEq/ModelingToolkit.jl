using ModelingToolkit, StaticArrays, LinearAlgebra
using OrdinaryDiffEq
using DiffEqBase, SparseArrays
using Test

using ModelingToolkit: value

# Define some variables
@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

# Define a differential equation
eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

ModelingToolkit.toexpr.(eqs)[1]
:(derivative(x(t), t) = σ * (y(t) - x(t))).args
de = ODESystem(eqs)

generate_function(de)

function test_diffeq_inference(name, sys, iv, dvs, ps)
    @testset "ODESystem construction: $name" begin
        @test isequal(independent_variable(sys),  value(iv))
        @test isempty(setdiff(Set(states(sys)), Set(value.(dvs))))
        @test isempty(setdiff(Set(parameters(sys)), Set(value.(ps))))
    end
end

test_diffeq_inference("standard", de, t, [x, y, z], [ρ, σ, β])
generate_function(de, [x,y,z], [σ,ρ,β])
jac_expr = generate_jacobian(de)
jac = calculate_jacobian(de)
jacfun = eval(jac_expr[2])
# iip
f = ODEFunction(de, [x,y,z], [σ,ρ,β])
du = zeros(3)
u  = collect(1:3)
p  = collect(4:6)
f(du, u, p, 0.1)
@test du == [4, 0, -16]
J = zeros(3, 3)
jacfun(J, u, p, t)
# oop
f = ODEFunction(de, [x,y,z], [σ,ρ,β])
du = @SArray zeros(3)
u  = SVector(1:3...)
p  = SVector(4:6...)
@test f(u, p, 0.1) === @SArray [4, 0, -16]

eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y*t,
       D(z) ~ x*y - β*z]
de = ODESystem(eqs) # This is broken
ModelingToolkit.calculate_tgrad(de)

tgrad_oop, tgrad_iip = eval.(ModelingToolkit.generate_tgrad(de))

@test tgrad_oop(u,p,t) == [0.0,-u[2],0.0]
du = zeros(3)
tgrad_iip(du,u,p,t)
@test du == [0.0,-u[2],0.0]

@testset "time-varying parameters" begin
    @parameters σ′(t-1)
    eqs = [D(x) ~ σ′*(y-x),
           D(y) ~ x*(ρ-z)-y,
           D(z) ~ x*y - β*z]
    de = ODESystem(eqs)
    test_diffeq_inference("global iv-varying", de, t, (x, y, z), (σ′, ρ, β))
    @test begin
        f = eval(generate_function(de, [x,y,z], [σ′,ρ,β])[2])
        du = [0.0,0.0,0.0]
        f(du, [1.0,2.0,3.0], [x->x+7,2,3], 5.0)
        du ≈ [11, -3, -7]
    end

    @parameters σ(..)
    eqs = [D(x) ~ σ(t-1)*(y-x),
           D(y) ~ x*(ρ-z)-y,
           D(z) ~ x*y - β*z]
    de = ODESystem(eqs)
    test_diffeq_inference("single internal iv-varying", de, t, (x, y, z), (σ(t-1), ρ, β))
    @test begin
        f = eval(generate_function(de, [x,y,z], [σ,ρ,β])[2])
        du = [0.0,0.0,0.0]
        f(du, [1.0,2.0,3.0], [x->x+7,2,3], 5.0)
        du ≈ [11, -3, -7]
    end

    eqs = [D(x) ~ x + 10σ(t-1) + 100σ(t-2) + 1000σ(t^2)]
    de = ODESystem(eqs)
    test_diffeq_inference("many internal iv-varying", de, t, (x,), (σ(t-2),σ(t^2), σ(t-1)))
    @test begin
        f = eval(generate_function(de, [x], [σ])[2])
        du = [0.0]
        f(du, [1.0], [t -> t + 2], 5.0)
        du ≈ [27561]
    end
end

# Conversion to first-order ODEs #17
D3 = Differential(t)^3
D2 = Differential(t)^2
@variables u(t) uˍtt(t) uˍt(t) xˍt(t)
eqs = [D3(u) ~ 2(D2(u)) + D(u) + D(x) + 1
       D2(x) ~ D(x) + 2]
de = ODESystem(eqs)
de1 = ode_order_lowering(de)
lowered_eqs = [D(uˍtt) ~ 2uˍtt + uˍt + xˍt + 1
               D(xˍt)  ~ xˍt + 2
               D(uˍt)  ~ uˍtt
               D(u)    ~ uˍt
               D(x)    ~ xˍt]

#@test de1 == ODESystem(lowered_eqs)

# issue #219
@test all(isequal.([ModelingToolkit.var_from_nested_derivative(eq.lhs)[1] for eq in de1.eqs], ODESystem(lowered_eqs).states))

test_diffeq_inference("first-order transform", de1, t, [uˍtt, xˍt, uˍt, u, x], [])
du = zeros(5)
ODEFunction(de1, [uˍtt, xˍt, uˍt, u, x], [])(du, ones(5), nothing, 0.1)
@test du == [5.0, 3.0, 1.0, 1.0, 1.0]

# Internal calculations
a = y - x
eqs = [D(x) ~ σ*a,
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]
de = ODESystem(eqs)
generate_function(de, [x,y,z], [σ,ρ,β])
jac = calculate_jacobian(de)
@test ModelingToolkit.jacobian_sparsity(de).colptr == sparse(jac).colptr
@test ModelingToolkit.jacobian_sparsity(de).rowval == sparse(jac).rowval

f = ODEFunction(de, [x,y,z], [σ,ρ,β])

D = Differential(t)
@parameters A B C
_x = y / C
eqs = [D(x) ~ -A*x,
       D(y) ~ A*x - B*_x]
de = ODESystem(eqs)
@test begin
    local f, du
    f = eval(generate_function(de, [x,y], [A,B,C])[2])
    du = [0.0,0.0]
    f(du, [1.0,2.0], [1,2,3], 0.0)
    du ≈ [-1, -1/3]
    f = eval(generate_function(de, [x,y], [A,B,C])[1])
    du ≈ f([1.0,2.0], [1,2,3], 0.0)
end

function lotka(u,p,t)
  x = u[1]
  y = u[2]
  [p[1]*x - p[2]*x*y,
  -p[3]*y + p[4]*x*y]
end

prob = ODEProblem(ODEFunction{false}(lotka),[1.0,1.0],(0.0,1.0),[1.5,1.0,3.0,1.0])
de = modelingtoolkitize(prob)
ODEFunction(de)(similar(prob.u0), prob.u0, prob.p, 0.1)

function lotka(du,u,p,t)
  x = u[1]
  y = u[2]
  du[1] = p[1]*x - p[2]*x*y
  du[2] = -p[3]*y + p[4]*x*y
end

prob = ODEProblem(lotka,[1.0,1.0],(0.0,1.0),[1.5,1.0,3.0,1.0])

de = modelingtoolkitize(prob)
ODEFunction(de)(similar(prob.u0), prob.u0, prob.p, 0.1)

# automatic state detection for DAEs
@parameters t k₁ k₂ k₃
@variables y₁(t) y₂(t) y₃(t)
D = Differential(t)
# reorder the system just to be a little spicier
eqs = [D(y₁) ~ -k₁*y₁+k₃*y₂*y₃,
       0     ~  y₁ + y₂ + y₃ - 1,
       D(y₂) ~  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃]
sys = ODESystem(eqs)
u0 = [y₁ => 1.0,
      y₂ => 0.0,
      y₃ => 0.0]
p  = [k₁ => 0.04,
      k₂ => 3e7,
      k₃ => 1e4]
p2 = (k₁ => 0.04,
      k₂ => 3e7,
      k₃ => 1e4)
tspan = (0.0,100000.0)
prob1 = ODEProblem(sys,u0,tspan,p)
prob12 = ODEProblem(sys,u0,tspan,[0.04,3e7,1e4])
prob13 = ODEProblem(sys,u0,tspan,(0.04,3e7,1e4))
prob14 = ODEProblem(sys,u0,tspan,p2)
prob2 = ODEProblem(sys,u0,tspan,p,jac=true)
prob3 = ODEProblem(sys,u0,tspan,p,jac=true,sparse=true)
for (prob, atol) in [(prob1, 1e-12), (prob2, 1e-12), (prob3, 1e-12)]
    sol = solve(prob, Rodas5())
    @test all(x->≈(sum(x), 1.0, atol=atol), sol.u)
end

@parameters t σ β
@variables x(t) y(t) z(t)
D = Differential(t)
eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x-β*y,
       x + z ~ y]
sys = ODESystem(eqs)
@test all(isequal.(states(sys), [x, y, z]))
@test all(isequal.(parameters(sys), [σ, β]))
@test equations(sys) == eqs

# issue 701
using ModelingToolkit
@parameters t a
@variables x(t)
D = Differential(t)
sys = ODESystem([D(x) ~ a])
@test sys.eqs[1].rhs isa Sym

# issue 708
@parameters t a
@variables x(t) y(t) z(t)
D = Differential(t)
sys = ODESystem([D(x) ~ y, 0 ~ x + z, 0 ~ x - y], t, [z, y, x], [])
sys2 = ode_order_lowering(sys)
M = ModelingToolkit.calculate_massmatrix(sys2)
@test M == Diagonal([1, 0, 0])

# issue #609
@variables t x1(t) x2(t)
D = Differential(t)

eqs = [
   D(x1) ~ -x1,
   0 ~ x1 - x2,
]
sys = ODESystem(eqs, t)
@test isequal(sys.iv, t)
@test isequal(states(sys), [x1, x2])
@test isempty(parameters(sys))
