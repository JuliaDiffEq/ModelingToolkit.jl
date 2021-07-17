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
struct JumpSystem{U <: ArrayPartition} <: AbstractTimeDependentSystem
    """
    The jumps of the system. Allowable types are `ConstantRateJump`,
    `VariableRateJump`, `MassActionJump`.
    """
    eqs::U
    """The independent variable, usually time."""
    iv::Any
    """The dependent variables, representing the state of the system.  Must not contain the independent variable."""
    states::Vector
    """The parameters of the system. Must not contain the independent variable."""
    ps::Vector
    observed::Vector{Equation}
    """The name of the system. . These are required to have unique names."""
    name::Symbol
    """The internal systems."""
    systems::Vector{JumpSystem}
    """
    defaults: The default values to use when initial conditions and/or
    parameters are not supplied in `ODEProblem`.
    """
    defaults::Dict
    """
    type: type of the system
    """
    connection_type::Any
    function JumpSystem{U}(ap::U, iv, states, ps, observed, name, systems, defaults, connection_type) where U <: ArrayPartition
        check_variables(states, iv)
        check_parameters(ps, iv)
        new{U}(ap, iv, states, ps, observed, name, systems, defaults, connection_type)
    end
end

function JumpSystem(eqs, iv, states, ps;
                    observed = Equation[],
                    systems = JumpSystem[],
                    default_u0=Dict(),
                    default_p=Dict(),
                    defaults=_merge(Dict(default_u0), Dict(default_p)),
                    name = gensym(:JumpSystem),
                    connection_type=nothing,
                    kwargs...)

    eqs = collect(eqs)
    sysnames = nameof.(systems)
    if length(unique(sysnames)) != length(sysnames)
        throw(ArgumentError("System names must be unique."))
    end
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
    if !(isempty(default_u0) && isempty(default_p))
        Base.depwarn("`default_u0` and `default_p` are deprecated. Use `defaults` instead.", :JumpSystem, force=true)
    end
    defaults = todict(defaults)
    defaults = Dict(value(k) => value(v) for (k, v) in pairs(defaults))

    states, ps = value.(states), value.(ps)
    collect_defaults!(defaults, states)
    collect_defaults!(defaults, ps)

    JumpSystem{typeof(ap)}(ap, value(iv), states, ps, observed, name, systems, defaults, connection_type)
end

function generate_rate_function(js::JumpSystem, rate)
    rf = build_function(rate, states(js), parameters(js),
                   get_iv(js),
                   conv = states_to_sym(states(js)),
                   expression=Val{true})
end
function add_integrator_header()
  integrator = gensym(:MTKIntegrator)

  expr -> Func([DestructuredArgs(expr.args, integrator, inds=[:u, :p, :t])], [], expr.body),
  expr -> Func([DestructuredArgs(expr.args, integrator, inds=[:u, :u, :p, :t])], [], expr.body)
end

function generate_affect_function(js::JumpSystem, affect, outputidxs)
    bf = build_function(map(x->x isa Equation ? x.rhs : x , affect), states(js),
                   parameters(js),
                   get_iv(js),
                   expression=Val{true},
                   wrap_code=add_integrator_header(),
                   outputidxs=outputidxs)[2]
end

function assemble_vrj(js, vrj, statetoid)
    rate   = @RuntimeGeneratedFunction(generate_rate_function(js, vrj.rate))
    outputvars = (value(affect.lhs) for affect in vrj.affect!)
    outputidxs = [statetoid[var] for var in outputvars]
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
    outputidxs = [statetoid[var] for var in outputvars]
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
        if !istree(spec) && _iszero(spec)
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
        !istree(spec) && _iszero(spec) && error("Net stoichiometry can not have a species labelled 0.")
        push!(ns, statetoid[spec] => stoich)
    end
    sort!(ns)
end

# assemble a numeric MassActionJump from a MT MassActionJump representing one rx.
function assemble_maj(maj::MassActionJump, statetoid, subber, invttype)
    rval = subber(maj.scaled_rates)
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
function DiffEqBase.DiscreteProblem(sys::JumpSystem, u0map, tspan::Union{Tuple,Nothing},
                                    parammap=DiffEqBase.NullParameters(); kwargs...)
    defs = defaults(sys)
    u0 = varmap_to_vars(u0map, states(sys); defaults=defs)
    p  = varmap_to_vars(parammap, parameters(sys); defaults=defs)
    f  = DiffEqBase.DISCRETE_INPLACE_DEFAULT
    df = DiscreteFunction{true,true}(f, syms=Symbol.(states(sys)))
    DiscreteProblem(df, u0, tspan, p; kwargs...)
end

"""
```julia
function DiffEqBase.DiscreteProblemExpr(sys::JumpSystem, u0map, tspan,
                                    parammap=DiffEqBase.NullParameters; kwargs...)
```

Generates a blank DiscreteProblem for a JumpSystem to utilize as its
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
function DiscreteProblemExpr(sys::JumpSystem, u0map, tspan::Union{Tuple,Nothing},
                                    parammap=DiffEqBase.NullParameters(); kwargs...)
    defs = defaults(sys)
    u0 = varmap_to_vars(u0map, states(sys); defaults=defs)
    p  = varmap_to_vars(parammap, parameters(sys); defaults=defs)
    # identity function to make syms works
    quote
        f  = DiffEqBase.DISCRETE_INPLACE_DEFAULT
        u0 = $u0
        p = $p
        tspan = $tspan
        df = DiscreteFunction{true,true}(f, syms=$(Symbol.(states(sys))))
        DiscreteProblem(df, u0, tspan, p)
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
    invttype  = prob.tspan[1] === nothing ? Float64 : typeof(1 / prob.tspan[2])

    # handling parameter substition and empty param vecs
    p = (prob.p isa DiffEqBase.NullParameters || prob.p === nothing) ? Num[] : prob.p
    parammap  = map((x,y)->Pair(x,y), parameters(js), p)
    subber    = substituter(parammap)

    majs = MassActionJump[assemble_maj(j, statetoid, subber, invttype) for j in eqs.x[1]]
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
function get_variables!(dep, jump::Union{ConstantRateJump,VariableRateJump}, variables)
    (jump.rate isa Symbolic) && get_variables!(dep, jump.rate, variables)
    dep
end

function get_variables!(dep, jump::MassActionJump, variables)
    sr = value(jump.scaled_rates)
    (sr isa Symbolic) && get_variables!(dep, sr, variables)
    for varasop in jump.reactant_stoch
        any(isequal(varasop[1]), variables) && push!(dep, varasop[1])
    end
    dep
end

### Functions to determine which states are modified by a given jump
function modified_states!(mstates, jump::Union{ConstantRateJump,VariableRateJump}, sts)
    for eq in jump.affect!
        st = eq.lhs
        any(isequal(st), sts) && push!(mstates, st)
    end
end

function modified_states!(mstates, jump::MassActionJump, sts)
    for (state,stoich) in jump.net_stoch
        any(isequal(state), sts) && push!(mstates, state)
    end
end
