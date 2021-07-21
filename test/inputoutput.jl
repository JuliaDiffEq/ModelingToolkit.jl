using ModelingToolkit, OrdinaryDiffEq, Symbolics, Test

@parameters t σ ρ β
@variables x(t) y(t) z(t) F(t) u(t)
D = Differential(t)

eqs = [D(x) ~ σ*(y-x) + F,
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

aliases = [u ~ x + y - z]
lorenz1 = ODESystem(eqs,pins=[F],observed=aliases,name=:lorenz1)
lorenz2 = ODESystem(eqs,pins=[F],observed=aliases,name=:lorenz2)

connections = [lorenz1.F ~ lorenz2.u,
               lorenz2.F ~ lorenz1.u]
connected = ODESystem(Equation[],t,[],[],observed=connections,systems=[lorenz1,lorenz2])

sys = connected

@variables lorenz1₊F lorenz2₊F
@test pins(connected) == Variable[lorenz1₊F, lorenz2₊F]
@test isequal(observed(connected),
              [connections...,
               lorenz1.u ~ lorenz1.x + lorenz1.y - lorenz1.z,
               lorenz2.u ~ lorenz2.x + lorenz2.y - lorenz2.z])

collapsed_eqs = [D(lorenz1.x) ~ (lorenz1.σ * (lorenz1.y - lorenz1.x) +
                                 (lorenz2.x + lorenz2.y - lorenz2.z)),
                 D(lorenz1.y) ~ lorenz1.x * (lorenz1.ρ - lorenz1.z) - lorenz1.y,
                 D(lorenz1.z) ~ lorenz1.x * lorenz1.y - (lorenz1.β * lorenz1.z),
                 D(lorenz2.x) ~ (lorenz2.σ * (lorenz2.y - lorenz2.x) +
                                 (lorenz1.x + lorenz1.y - lorenz1.z)),
                 D(lorenz2.y) ~ lorenz2.x * (lorenz2.ρ - lorenz2.z) - lorenz2.y,
                 D(lorenz2.z) ~ lorenz2.x * lorenz2.y - (lorenz2.β * lorenz2.z)]

simplifyeqs(eqs) = Equation.((x->x.lhs).(eqs), simplify.((x->x.rhs).(eqs)))

@test isequal(simplifyeqs(equations(connected)), simplifyeqs(collapsed_eqs))

# Test if variables indicated to be input/output are not eliminated by structural_simplify

@variables x [input=true]
@test hasmetadata(x, Symbolics.option_to_metadata_type(Val(:input)))
@test getmetadata(x, Symbolics.option_to_metadata_type(Val(:input))) == true
@test !hasmetadata(x, Symbolics.option_to_metadata_type(Val(:output)))
@test_throws KeyError getmetadata(x, Symbolics.option_to_metadata_type(Val(:output)))

@variables y [output=true]
@test hasmetadata(y, Symbolics.option_to_metadata_type(Val(:output)))
@test getmetadata(y, Symbolics.option_to_metadata_type(Val(:output))) == true
@test !hasmetadata(y, Symbolics.option_to_metadata_type(Val(:input)))
@test_throws KeyError getmetadata(y, Symbolics.option_to_metadata_type(Val(:input)))

@parameters t σ ρ β
@variables x(t) y(t) z(t) a(t) u(t) F(t)
D = Differential(t)

eqs = [
       D(x) ~ σ*(y-x)
       D(y) ~ x*(ρ-z)-y + β
       0 ~ z - x + y
       0 ~ a + z
       u ~ z + a
      ]

lorenz1 = ODESystem(eqs,t,name=:lorenz1)

lorenz1_aliased = structural_simplify(lorenz1)

reduced_eqs = [
               D(x) ~ σ*(y - x)
               D(y) ~ β + x*(ρ - (x - y)) - y
              ]

@info states(lorenz1)
@info states(lorenz1_aliased)