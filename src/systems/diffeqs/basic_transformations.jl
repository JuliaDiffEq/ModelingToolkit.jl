"""
$(TYPEDSIGNATURES)

Generates the Liouville transformed set of ODEs, which is the original
ODE system with a new variable `trJ` appended, corresponding to the
-tr(Jacobian). This variable is used for properties like uncertainty
propagation from a given initial distribution density.

For example, if ``u'=p*u`` and `p` follows a probability distribution
``f(p)``, then the probability density of a future value with a given
choice of ``p`` is computed by setting the inital `trJ = f(p)`, and
the final value of `trJ` is the probability of ``u(t)``.

Example:

```julia
using ModelingToolkit, OrdinaryDiffEq, Test

@parameters t α β γ δ
@variables x(t) y(t)
D = Differential(t)

eqs = [D(x) ~ α*x - β*x*y,
       D(y) ~ -δ*y + γ*x*y]

sys = ODESystem(eqs)
sys2 = liouville_transform(sys)
@variables trJ

u0 = [x => 1.0,
      y => 1.0,
      trJ => 1.0]

prob = ODEProblem(sys2,u0,tspan,p)
sol = solve(prob,Tsit5())
```

Where `sol[3,:]` is the evolution of `trJ` over time.

Sources:

Probabilistic Robustness Analysis of F-16 Controller Performance: An
Optimal Transport Approach

Abhishek Halder, Kooktae Lee, and Raktim Bhattacharya
https://abhishekhalder.bitbucket.io/F16ACC2013Final.pdf
"""
function liouville_transform(sys)
      t = independent_variable(sys)
      @variables trJ
      D = ModelingToolkit.Differential(t)
      neweq = D(trJ) ~ trJ*-tr(calculate_jacobian(sys))
      neweqs = [equations(sys);neweq]
      vars = [states(sys);trJ]
      ODESystem(neweqs,t,vars,parameters(sys))
end








"""
$(TYPEDSIGNATURES)

Generates the set of ODEs after change of variables.


Example:

```julia
using ModelingToolkit, OrdinaryDiffEq, Test

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
forward_subs  = [exp(x) => z]
backward_subs = [x => log(z)]
new_sys = changeofvariables(sys, forward_subs, backward_subs)
@test equations(new_sys)[1] == (D(z) ~ α*z*log(z))

new_prob = ODEProblem(new_sys, [], tspan, p)
new_sol = solve(new_prob, Tsit5())

@test isapprox(new_sol[x][end], sol[x][end], atol=1e-4)
```

"""
function changeofvariables(sys::ODESystem, forward_subs, backward_subs; simplify=false, t0=missing)
    t = independent_variable(sys)

    old_vars = first.(backward_subs)
    new_vars = last.(forward_subs)
    kept_vars = setdiff(states(sys), old_vars)
    rhs = [eq.rhs for eq in equations(sys)]

    # use: dz/dt = ∂z/∂x dx/dt + ∂z/∂t
    dzdt = Symbolics.derivative( first.(forward_subs), t )
    new_eqs = Equation[]
    for (new_var, ex) in zip(new_vars, dzdt)
        for ode_eq in equations(sys)
            ex = substitute(ex, ode_eq.lhs => ode_eq.rhs)
        end
        ex = substitute(ex, Dict(forward_subs))
        ex = substitute(ex, Dict(backward_subs))
        if simplify
            ex = Symbolics.simplify(ex, expand=true)
        end
        push!(new_eqs, Differential(t)(new_var) ~ ex)
    end

    defs = get_defaults(sys)
    new_defs = Dict()
    for f_sub in forward_subs
        #TODO call value(...)?
        ex = substitute(first(f_sub), defs)
        if !ismissing(t0)
            ex = substitute(ex, t => t0)
        end
        new_defs[last(f_sub)] = ex
    end
    return ODESystem(new_eqs;
                        defaults=new_defs,
                        observed=vcat(observed(sys),first.(backward_subs) .~ last.(backward_subs))
                        )
end
