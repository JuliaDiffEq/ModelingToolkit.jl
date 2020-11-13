JumpType = Union{VariableRateJump, ConstantRateJump, MassActionJump}

"""
$(TYPEDEF)

A system of jump processes.

# Fields
$(FIELDS)

# Example

```julia
using ModelingToolkit

@parameters β γ t
@variables S I R
rate₁   = β*S*I
affect₁ = [S ~ S - 1, I ~ I + 1]
rate₂   = γ*I
affect₂ = [I ~ I - 1, R ~ R + 1]
j₁      = ConstantRateJump(rate₁,affect₁)
j₂      = ConstantRateJump(rate₂,affect₂)
j₃      = MassActionJump(2*β+γ, [R => 1], [S => 1, R => -1])
js      = JumpSystem([j₁,j₂,j₃], t, [S,I,R], [β,γ])
```
"""
struct JumpSystem{U <: ArrayPartition} <: AbstractSystem
    """
    The jumps of the system. Allowable types are `ConstantRateJump`,
    `VariableRateJump`, `MassActionJump`.
    """
    eqs::U
    """The independent variable, usually time."""
    iv::Any
    """The dependent variables, representing the state of the system."""
    states::Vector
    """The parameters of the system."""
    ps::Vector
    pins::Vector
    observed::Vector{Equation}
    """The name of the system."""
    name::Symbol
    """The internal systems."""
    systems::Vector{JumpSystem}
end

function JumpSystem(eqs, iv, states, ps;
                    pins = [],
                    observed = Equation[],
                    systems = JumpSystem[],
                    name = gensym(:JumpSystem))

    ap = ArrayPartition(MassActionJump[], ConstantRateJump[], VariableRateJump[])
    for eq in eqs
        if eq isa MassActionJump
            push!(ap.x[1], eq)
        elseif eq isa ConstantRateJump
            push!(ap.x[2], eq)
        elseif eq isa VariableRateJump
            push!(ap.x[3], eq)
        else
            error("JumpSystem equations must contain MassActionJumps, ConstantRateJumps, or VariableRateJumps.")
        end
    end

    JumpSystem{typeof(ap)}(ap, value(iv), value.(states), value.(ps), pins, observed, name, systems)
end

function generate_rate_function(js, rate)
    build_function(rate, states(js), parameters(js),
                   independent_variable(js),
                   conv = states_to_sym(states(js)),
                   expression=Val{true})
end

function generate_affect_function(js, affect, outputidxs)
    build_function(affect, states(js),
                   parameters(js),
                   conv = states_to_sym(states(js)),
                   independent_variable(js),
                   expression=Val{true},
                   headerfun=add_integrator_header,
                   outputidxs=outputidxs)[2]
end

function assemble_vrj(js, vrj, statetoid)
    rate   = @RuntimeGeneratedFunction(generate_rate_function(js, vrj.rate))
    outputvars = (value(affect.lhs) for affect in vrj.affect!)
    outputidxs = ((statetoid[var] for var in outputvars)...,)
    affect = @RuntimeGeneratedFunction(generate_affect_function(js, vrj.affect!, outputidxs))
    VariableRateJump(rate, affect)
end

function assemble_vrj_expr(js, vrj, statetoid)
    rate   = generate_rate_function(js, vrj.rate)
    outputvars = (value(affect.lhs) for affect in vrj.affect!)
    outputidxs = ((statetoid[var] for var in outputvars)...,)
    affect = generate_affect_function(js, vrj.affect!, outputidxs)
    quote
        rate = $rate
        affect = $affect
        VariableRateJump(rate, affect)
    end
end

function assemble_crj(js, crj, statetoid)
    rate   = @RuntimeGeneratedFunction(generate_rate_function(js, crj.rate))
    outputvars = (value(affect.lhs) for affect in crj.affect!)
    outputidxs = ((statetoid[var] for var in outputvars)...,)
    affect = @RuntimeGeneratedFunction(generate_affect_function(js, crj.affect!, outputidxs))
    ConstantRateJump(rate, affect)
end

function assemble_crj_expr(js, crj, statetoid)
    rate   = generate_rate_function(js, crj.rate)
    outputvars = (value(affect.lhs) for affect in crj.affect!)
    outputidxs = ((statetoid[var] for var in outputvars)...,)
    affect = generate_affect_function(js, crj.affect!, outputidxs)
    quote
        rate = $rate
        affect = $affect
        ConstantRateJump(rate, affect)
    end
end

function numericrstoich(mtrs::Vector{Pair{V,W}}, statetoid) where {V,W}
    rs = Vector{Pair{Int,W}}()
    for (spec,stoich) in mtrs
        if !(spec isa Term) && _iszero(spec)
            push!(rs, 0 => stoich)
        else
            push!(rs, statetoid[value(spec)] => stoich)
        end
    end
    sort!(rs)
    rs
end

function numericnstoich(mtrs::Vector{Pair{V,W}}, statetoid) where {V,W}
    ns = Vector{Pair{Int,W}}()
    for (spec,stoich) in mtrs
        !(spec isa Term) && _iszero(spec) && error("Net stoichiometry can not have a species labelled 0.")
        push!(ns, statetoid[spec] => stoich)
    end
    sort!(ns)
end

# assemble a numeric MassActionJump from a MT MassActionJump representing one rx.
function assemble_maj(maj::MassActionJump, statetoid, parammap, invttype)
    rval = substitute(maj.scaled_rates, parammap)
    rs   = numericrstoich(maj.reactant_stoch, statetoid)
    ns   = numericnstoich(maj.net_stoch, statetoid)
    maj  = MassActionJump(convert(invttype, value(rval)), rs, ns, scale_rates = false)
    maj
end

# For MassActionJumps that contain many reactions
# function assemble_maj(maj::MassActionJump{U,V,W}, statetoid, subber,
#                       invttype) where {U <: AbstractVector,V,W}
#     rval = [convert(invttype,numericrate(sr, subber)) for sr in maj.scaled_rates]
#     rs   = [numericrstoich(rs, statetoid) for rs in maj.reactant_stoch]
#     ns   = [numericnstoich(ns, statetoid) for ns in maj.net_stoch]
#     maj  = MassActionJump(rval, rs, ns, scale_rates = false)
#     maj
# end
"""
```julia
function DiffEqBase.DiscreteProblem(sys::JumpSystem, u0map, tspan,
                                    parammap=DiffEqBase.NullParameters; kwargs...)
```

Generates a blank DiscreteProblem for a pure jump JumpSystem to utilize as
its `prob.prob`. This is used in the case where there are no ODEs
and no SDEs associated with the system.

Continuing the example from the [`JumpSystem`](@ref) definition:
```julia
using DiffEqBase, DiffEqJump
u₀map = [S => 999, I => 1, R => 0]
parammap = [β => .1/1000, γ => .01]
tspan = (0.0, 250.0)
dprob = DiscreteProblem(js, u₀map, tspan, parammap)
```
"""
function DiffEqBase.DiscreteProblem(sys::JumpSystem, u0map, tspan::Tuple,
                                    parammap=DiffEqBase.NullParameters(); kwargs...)

    (u0map isa AbstractVector) || error("For DiscreteProblems u0map must be an AbstractVector.")
    u0d = Dict( value(u[1]) => u[2] for u in u0map)
    u0 = [u0d[u] for u in states(sys)]
    if parammap != DiffEqBase.NullParameters()
        (parammap isa AbstractVector) || error("For DiscreteProblems parammap must be an AbstractVector.")
        pd  = Dict( value(u[1]) => u[2] for u in parammap)
        p  = [pd[u] for u in parameters(sys)]
    else
        p = parammap
    end
    f  = DiffEqBase.DISCRETE_INPLACE_DEFAULT
    sts = states(sys)
    df = DiscreteFunction{true,true}(f, syms=tosymbol.(sts, states=sts, escape=false))
    DiscreteProblem(df, u0, tspan, p; kwargs...)
end

"""
```julia
function DiffEqBase.DiscreteProblemExpr(sys::JumpSystem, u0map, tspan,
                                    parammap=DiffEqBase.NullParameters; kwargs...)
```

Generates a black DiscreteProblem for a JumpSystem to utilize as its
solving `prob.prob`. This is used in the case where there are no ODEs
and no SDEs associated with the system.

Continuing the example from the [`JumpSystem`](@ref) definition:
```julia
using DiffEqBase, DiffEqJump
u₀map = [S => 999, I => 1, R => 0]
parammap = [β => .1/1000, γ => .01]
tspan = (0.0, 250.0)
dprob = DiscreteProblem(js, u₀map, tspan, parammap)
```
"""
function DiscreteProblemExpr(sys::JumpSystem, u0map, tspan::Tuple,
                                    parammap=DiffEqBase.NullParameters(); kwargs...)
    u0 = varmap_to_vars(u0map, states(sys))
    p  = varmap_to_vars(parammap, parameters(sys))
    # identity function to make syms works
    sts = states(sys)
    quote
        f  = DiffEqBase.DISCRETE_INPLACE_DEFAULT
        u0 = $u0
        p = $p
        tspan = $tspan
        df = DiscreteFunction{true,true}(f, syms=$(tosymbol.(sts, states=sts, escape=false)))
        DiscreteProblem(df, u0, tspan, p; kwargs...)
    end
end

"""
```julia
function DiffEqBase.JumpProblem(js::JumpSystem, prob, aggregator; kwargs...)
```

Generates a JumpProblem from a JumpSystem.

Continuing the example from the [`DiscreteProblem`](@ref) definition:
```julia
jprob = JumpProblem(js, dprob, Direct())
sol = solve(jprob, SSAStepper())
```
"""
function DiffEqJump.JumpProblem(js::JumpSystem, prob, aggregator; kwargs...)

    statetoid = Dict(value(state) => i for (i,state) in enumerate(states(js)))
    eqs       = equations(js)
    invttype  = typeof(1 / prob.tspan[2])

    # handling parameter substition and empty param vecs
    p = (prob.p == DiffEqBase.NullParameters()) ? Num[] : prob.p
    parammap  = Dict(value(x) => value(y) for (x, y) in zip(parameters(js), p))

    majs = MassActionJump[assemble_maj(j, statetoid, parammap, invttype) for j in eqs.x[1]]
    crjs = ConstantRateJump[assemble_crj(js, j, statetoid) for j in eqs.x[2]]
    vrjs = VariableRateJump[assemble_vrj(js, j, statetoid) for j in eqs.x[3]]
    ((prob isa DiscreteProblem) && !isempty(vrjs)) && error("Use continuous problems such as an ODEProblem or a SDEProblem with VariableRateJumps")
    jset = JumpSet(Tuple(vrjs), Tuple(crjs), nothing, isempty(majs) ? nothing : majs)

    if needs_vartojumps_map(aggregator) || needs_depgraph(aggregator)
        jdeps = asgraph(js)
        vdeps = variable_dependencies(js)
        vtoj = jdeps.badjlist
        jtov = vdeps.badjlist
        jtoj = needs_depgraph(aggregator) ? eqeq_dependencies(jdeps, vdeps).fadjlist : nothing
    else
        vtoj = nothing; jtov = nothing; jtoj = nothing
    end

    JumpProblem(prob, aggregator, jset; dep_graph=jtoj, vartojumps_map=vtoj, jumptovars_map=jtov, kwargs...)
end


### Functions to determine which states a jump depends on
function get_variables!(dep, jump::Union{ConstantRateJump,VariableRateJump}, variables::AbstractSet)
    (jump.rate isa Symbolic) && get_variables!(dep, jump.rate, variables)
    dep
end

function get_variables!(dep, jump::MassActionJump, variables::AbstractSet)
    sr = value(jump.scaled_rates)
    (sr isa Symbolic) && get_variables!(dep, sr, variables)
    for varasop in jump.reactant_stoch
        varasop[1] in variables && push!(dep, varasop[1])
    end
    dep
end

### Functions to determine which states are modified by a given jump
function modified_states!(mstates, jump::Union{ConstantRateJump,VariableRateJump}, sts::AbstractSet)
    for eq in jump.affect!
        st = eq.lhs
        st in sts && push!(mstates, st)
    end
end

function modified_states!(mstates, jump::MassActionJump, sts::AbstractSet)
    for (state,stoich) in jump.net_stoch
        state in sts && push!(mstates, state)
    end
end
