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
function liouville_transform(sys::AbstractODESystem)
      t = get_iv(sys)
      @variables trJ
      D = ModelingToolkit.Differential(t)
      neweq = D(trJ) ~ trJ*-tr(calculate_jacobian(sys))
      neweqs = [equations(sys);neweq]
      vars = [states(sys);trJ]
      ODESystem(neweqs,t,vars,parameters(sys))
end
