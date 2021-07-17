# The AbstractSystem Interface

## Overview

The `AbstractSystem` interface is the core of the system level of ModelingToolkit.jl.
It establishes a common set of functionality that is used between systems
from ODEs and chemical reactions, allowing users to have a common framework for
model manipulation and compilation.

### Subtypes

There are three immediate subtypes of `AbstractSystem`, classified by how many independent variables each type has:
* `AbstractTimeIndependentSystem`: has no independent variable (eg: `NonlinearSystem`)
* `AbstractTimeDependentSystem`: has a single independent variable (eg: `ODESystem`)
* `AbstractMultivariateSystem`: may have multiple independent variables (eg: `PDESystem`)

## Constructors and Naming

The `AbstractSystem` interface has a consistent method for constructing systems.
Generally it follows the order of:

1. Equations
2. Independent Variables
3. Dependent Variables (or States)
4. Parameters

All other pieces are handled via keyword arguments. `AbstractSystem`s share the
same keyword arguments, which are:

- `system`: This is used for specifying subsystems for hierarchical modeling with
  reusable components. For more information, see the [components page](@ref components)
- Defaults: Keyword arguments like `defaults` are used for specifying default
  values which are used. If a value is not given at the `SciMLProblem` construction
  time, its numerical value will be the default.

## Composition and Accessor Functions

Each `AbstractSystem` has lists of variables in context, such as distinguishing
parameters vs states. In addition, an `AbstractSystem` also can hold other
`AbstractSystem` types. Direct accessing of the values, such as `sys.states`,
gives the immediate list, while the accessor functions `states(sys)` gives the
total set, which includes that of all systems held inside.

The values which are common to all `AbstractSystem`s are:

- `equations(sys)`: All equations that define the system and its subsystems.
- `states(sys)`: All the states in the system and its subsystems.
- `parameters(sys)`: All parameters of the system and its subsystems.
- `nameof(sys)`: The name of the current-level system.
- `get_eqs(sys)`: Equations that define the current-level system.
- `get_states(sys)`: States that are in the current-level system.
- `get_ps(sys)`: Parameters that are in the current-level system.
- `get_systems(sys)`: Subsystems of the current-level system.

Optionally, a system could have:

- `observed(sys)`: All observed equations of the system and its subsystems.
- `get_observed(sys)`: Observed equations of the current-level system.
- `get_defaults(sys)`: A `Dict` that maps variables into their default values.
- `independent_variables(sys)`: The independent variables of a system.
- `get_noiseeqs(sys)`: Noise equations of the current-level system.

Note that if you know a system is an `AbstractTimeDependentSystem` you could use `get_iv` to get the 
unique independent variable directly, rather than using `independenent_variables(sys)[1]`, which is clunky and may cause problems if `sys` is an `AbstractMultivariateSystem` because there may be more than one independent variable. `AbstractTimeIndependentSystem`s do not have a method `get_iv`, and `independent_variables(sys)` will return a size-zero result for such. For an `AbstractMultivariateSystem`, `get_ivs` is equivalent.

A system could also have caches:

- `get_jac(sys)`: The Jacobian of a system.
- `get_tgrad(sys)`: The gradient with respect to time of a system.

## Transformations

Transformations are functions which send a valid `AbstractSystem` definition to
another `AbstractSystem`. These are passes, like optimizations (e.g., Block-Lower
Triangle transformations), or changes to the representation, which allow for
alternative numerical methods to be utilized on the model (e.g., DAE index reduction).

## Analyses

Analyses are functions on a system which return information about the corresponding
properties, like whether its parameters are structurally identifiable, or whether
it's linear.

## Function Calculation and Generation

The calculation and generation functions allow for calculating additional
quantities to enhance the numerical methods applied to the resulting system.
The calculations, like `calculate_jacobian`, generate ModelingToolkit IR for
the Jacobian of the system, while the generations, like `generate_jacobian`,
generate compiled output for the numerical solvers by applying `build_function`
to the generated code. Additionally, many systems have function-type outputs,
which cobble together the generation functionality for a system, for example,
`ODEFunction` can be used to generate a DifferentialEquations-based `ODEFunction`
with compiled version of the ODE itself, the Jacobian, the mass matrix, etc.

Below are the possible calculation and generation functions:

```@docs
calculate_tgrad
calculate_gradient
calculate_jacobian
calculate_factorized_W
calculate_hessian
generate_tgrad
generate_gradient
generate_jacobian
generate_factorized_W
generate_hessian
```

Additionally, `jacobian_sparsity(sys)` and `hessian_sparsity(sys)`
exist on the appropriate systems for fast generation of the sparsity
patterns via an abstract interpretation without requiring differentiation.

## Problem Constructors

At the end, the system types have `DEProblem` constructors, like `ODEProblem`,
which allow for directly generating the problem types required for numerical
methods. The first argument is always the `AbstractSystem`, and the proceeding
arguments match the argument order of their original constructors. Whenever an
array would normally be provided, such as `u0` the initial condition of an
`ODEProblem`, it is instead replaced with a variable map, i.e., an array of
pairs `var=>value`, which allows the user to designate the values without having
to know the order that ModelingToolkit is internally using.

For the value maps, the parameters are allowed to be functions of each other,
and value maps of states can be functions of the parameters, i.e. you can do:

```
u0 = [
  lorenz1.x => 2.0
  lorenz2.x => lorenz1.x * lorenz1.p
]
```

## Default Value Handling

The `AbstractSystem` types allow for specifying default values, for example
`defaults` inside of them. At problem construction time, these values are merged
into the value maps, where for any repeats the value maps override the default.
In addition, defaults of a higher level in the system override the defaults of
a lower level in the system.
