# ReactionSystem

A `ReactionSystem` represents a system of chemical reactions. Conversions are
provided to generate corresponding chemical reaction ODE models, chemical
Langevin equation SDE models, and stochastic chemical kinetics jump process
models. As a simple example, the code below creates a SIR model, and solves
the corresponding ODE, SDE, and jump process models.

```julia
using ModelingToolkit, OrdinaryDiffEq, StochasticDiffEq, DiffEqJump
@parameters β γ t
@variables S(t) I(t) R(t)

rxs = [Reaction(β, [S,I], [I], [1,1], [2])
       Reaction(γ, [I], [R])]
rs  = ReactionSystem(rxs, t, [S,I,R], [β,γ])

u₀map    = [S => 999.0, I => 1.0, R => 0.0]
parammap = [β => 1/10000, γ => 0.01]
tspan    = (0.0, 250.0)

# solve as ODEs
odesys = convert(ODESystem, rs)
oprob = ODEProblem(odesys, u₀map, tspan, parammap)
sol = solve(oprob, Tsit5())

# solve as SDEs
sdesys = convert(SDESystem, rs)
sprob = SDEProblem(sdesys, u₀map, tspan, parammap)
sol = solve(sprob, EM(), dt=.01)

# solve as jump process
jumpsys = convert(JumpSystem, rs)
u₀map    = [S => 999, I => 1, R => 0]
dprob = DiscreteProblem(jumpsys, u₀map, tspan, parammap)
jprob = JumpProblem(jumpsys, dprob, Direct())
sol = solve(jprob, SSAStepper())
```

## System Constructors

```@docs
Reaction
ReactionSystem
```

## Composition and Accessor Functions

- `get_eqs(sys)` or `equations(sys)`: The reactions that define the system.
- `get_states(sys)` or `states(sys)`: The set of chemical species in the system.
- `get_ps(sys)` or `parameters(sys)`: The parameters of the system.
- `get_iv(sys)`: The independent variable of the
  reaction system, usually time.

## Query Functions
```@docs
oderatelaw
jumpratelaw
ismassaction
```

## Transformations

```@docs
Base.convert
structural_simplify
```

## Analyses
