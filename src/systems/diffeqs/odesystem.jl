"""
$(TYPEDEF)

A system of ordinary differential equations.

# Fields
$(FIELDS)

# Example

```julia
using ModelingToolkit

@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

de = ODESystem(eqs,t,[x,y,z],[σ,ρ,β])
```
"""
struct ODESystem <: AbstractODESystem
    """The ODEs defining the system."""
    eqs::Vector{Equation}
    """Independent variable."""
    iv::Sym
    """Dependent (state) variables. Must not contain the independent variable."""
    states::Vector
    """Parameter variables. Must not contain the independent variable."""
    ps::Vector
    """Control parameters (some subset of `ps`)."""
    ctrls::Vector
    """Observed states."""
    observed::Vector{Equation}
    """
    Time-derivative matrix. Note: this field will not be defined until
    [`calculate_tgrad`](@ref) is called on the system.
    """
    tgrad::RefValue{Vector{Num}}
    """
    Jacobian matrix. Note: this field will not be defined until
    [`calculate_jacobian`](@ref) is called on the system.
    """
    jac::RefValue{Any}
    """
    Control Jacobian matrix. Note: this field will not be defined until
    [`calculate_control_jacobian`](@ref) is called on the system.
    """
    ctrl_jac::RefValue{Any}
    """
    `Wfact` matrix. Note: this field will not be defined until
    [`generate_factorized_W`](@ref) is called on the system.
    """
    Wfact::RefValue{Matrix{Num}}
    """
    `Wfact_t` matrix. Note: this field will not be defined until
    [`generate_factorized_W`](@ref) is called on the system.
    """
    Wfact_t::RefValue{Matrix{Num}}
    """
    Name: the name of the system
    """
    name::Symbol
    """
    systems: The internal systems. These are required to have unique names.
    """
    systems::Vector{ODESystem}
    """
    defaults: The default values to use when initial conditions and/or
    parameters are not supplied in `ODEProblem`.
    """
    defaults::Dict
    """
    structure: structural information of the system
    """
    structure::Any
    """
    type: type of the system
    """
    connection_type::Any

    function ODESystem(deqs, iv, dvs, ps, ctrls, observed, tgrad, jac, ctrl_jac, Wfact, Wfact_t, name, systems, defaults, structure, connection_type)
        check_variables(dvs,iv)
        check_parameters(ps,iv)
        check_equations(deqs,iv)
        new(deqs, iv, dvs, ps, ctrls, observed, tgrad, jac, ctrl_jac, Wfact, Wfact_t, name, systems, defaults, structure, connection_type)
    end
end

function ODESystem(
                   deqs::AbstractVector{<:Equation}, iv, dvs, ps;
                   controls  = Num[],
                   observed = Num[],
                   systems = ODESystem[],
                   name=gensym(:ODESystem),
                   default_u0=Dict(),
                   default_p=Dict(),
                   defaults=_merge(Dict(default_u0), Dict(default_p)),
                   connection_type=nothing,
                  )
    deqs = collect(deqs)
    @assert all(control -> any(isequal.(control, ps)), controls) "All controls must also be parameters."

    iv′ = value(scalarize(iv))
    dvs′ = value.(scalarize(dvs))
    ps′ = value.(scalarize(ps))
    ctrl′ = value.(scalarize(controls))

    if !(isempty(default_u0) && isempty(default_p))
        Base.depwarn("`default_u0` and `default_p` are deprecated. Use `defaults` instead.", :ODESystem, force=true)
    end
    defaults = todict(defaults)
    defaults = Dict(value(k) => value(v) for (k, v) in pairs(defaults))

    iv′ = value(scalarize(iv))
    dvs′ = value.(scalarize(dvs))
    ps′ = value.(scalarize(ps))

    collect_defaults!(defaults, dvs′)
    collect_defaults!(defaults, ps′)

    tgrad = RefValue(Vector{Num}(undef, 0))
    jac = RefValue{Any}(Matrix{Num}(undef, 0, 0))
    ctrl_jac = RefValue{Any}(Matrix{Num}(undef, 0, 0))
    Wfact   = RefValue(Matrix{Num}(undef, 0, 0))
    Wfact_t = RefValue(Matrix{Num}(undef, 0, 0))
    sysnames = nameof.(systems)
    if length(unique(sysnames)) != length(sysnames)
        throw(ArgumentError("System names must be unique."))
    end
    ODESystem(deqs, iv′, dvs′, ps′, ctrl′, observed, tgrad, jac, ctrl_jac, Wfact, Wfact_t, name, systems, defaults, nothing, connection_type)
end

vars(x::Sym) = Set([x])
vars(exprs::Symbolic) = vars([exprs])
vars(exprs) = foldl(vars!, exprs; init = Set())
vars!(vars, eq::Equation) = (vars!(vars, eq.lhs); vars!(vars, eq.rhs); vars)
function vars!(vars, O)
    if isa(O, Sym)
        return push!(vars, O)
    end
    !istree(O) && return vars

    operation(O) isa Differential && return push!(vars, O)

    if operation(O) === (getindex) &&
        first(arguments(O)) isa Symbolic

        return push!(vars, O)
    end

    symtype(operation(O)) <: FnType && push!(vars, O)
    for arg in arguments(O)
        vars!(vars, arg)
    end

    return vars
end

find_derivatives!(vars, expr::Equation, f=identity) = (find_derivatives!(vars, expr.lhs, f); find_derivatives!(vars, expr.rhs, f); vars)
function find_derivatives!(vars, expr, f)
    !istree(O) && return vars
    operation(O) isa Differential && push!(vars, f(O))
    for arg in arguments(O)
        vars!(vars, arg)
    end
    return vars
end

function ODESystem(eqs, iv=nothing; kwargs...)
    eqs = collect(eqs)
    # NOTE: this assumes that the order of algebric equations doesn't matter
    diffvars = OrderedSet()
    allstates = OrderedSet()
    ps = OrderedSet()
    # reorder equations such that it is in the form of `diffeq, algeeq`
    diffeq = Equation[]
    algeeq = Equation[]
    # initial loop for finding `iv`
    if iv === nothing
        for eq in eqs
            if !(eq.lhs isa Number) # assume eq.lhs is either Differential or Number
                iv = iv_from_nested_derivative(eq.lhs)
                break
            end
        end
    end
    iv = value(iv)
    iv === nothing && throw(ArgumentError("Please pass in independent variables."))
    for eq in eqs
        collect_vars!(allstates, ps, eq.lhs, iv)
        collect_vars!(allstates, ps, eq.rhs, iv)
        if isdiffeq(eq)
            diffvar, _ = var_from_nested_derivative(eq.lhs)
            isequal(iv, iv_from_nested_derivative(eq.lhs)) || throw(ArgumentError("An ODESystem can only have one independent variable."))
            diffvar in diffvars && throw(ArgumentError("The differential variable $diffvar is not unique in the system of equations."))
            push!(diffvars, diffvar)
            push!(diffeq, eq)
        else
            push!(algeeq, eq)
        end
    end
    algevars = setdiff(allstates, diffvars)
    # the orders here are very important!
    return ODESystem(append!(diffeq, algeeq), iv, vcat(collect(diffvars), collect(algevars)), ps; kwargs...)
end

function collect_vars!(states, parameters, expr, iv)
    if expr isa Sym
        collect_var!(states, parameters, expr, iv)
    else
        for var in vars(expr)
            if istree(var) && operation(var) isa Differential
                var, _ = var_from_nested_derivative(var)
            end
            collect_var!(states, parameters, var, iv)
        end
    end
    return nothing
end

function collect_var!(states, parameters, var, iv)
    isequal(var, iv) && return nothing
    if isparameter(var) || (istree(var) && isparameter(operation(var)))
        push!(parameters, var)
    else
        push!(states, var)
    end
    return nothing
end

# NOTE: equality does not check cached Jacobian
function Base.:(==)(sys1::ODESystem, sys2::ODESystem)
    iv1 = get_iv(sys1)
    iv2 = get_iv(sys2)
    isequal(iv1, iv2) &&
    _eq_unordered(get_eqs(sys1), get_eqs(sys2)) &&
    _eq_unordered(get_states(sys1), get_states(sys2)) &&
    _eq_unordered(get_ps(sys1), get_ps(sys2)) &&
    all(s1 == s2 for (s1, s2) in zip(get_systems(sys1), get_systems(sys2)))
end

function flatten(sys::ODESystem)
    systems = get_systems(sys)
    if isempty(systems)
        return sys
    else
        return ODESystem(
                         equations(sys),
                         get_iv(sys),
                         states(sys),
                         parameters(sys),
                         observed=observed(sys),
                         defaults=defaults(sys),
                         name=nameof(sys),
                        )
    end
end

ODESystem(eq::Equation, args...; kwargs...) = ODESystem([eq], args...; kwargs...)

"""
$(SIGNATURES)

Build the observed function assuming the observed equations are all explicit,
i.e. there are no cycles.
"""
function build_explicit_observed_function(
        sys, syms;
        expression=false,
        output_type=Array,
        checkbounds=true)

    if (isscalar = !(syms isa Vector))
        syms = [syms]
    end
    syms = value.(syms)

    obs = observed(sys)
    observed_idx = Dict(map(x->x.lhs, obs) .=> 1:length(obs))
    output = similar(syms, Any)
    # FIXME: this is a rather rough estimate of dependencies.
    maxidx = 0
    for (i, s) in enumerate(syms)
        idx = observed_idx[s]
        idx > maxidx && (maxidx = idx)
        output[i] = obs[idx].rhs
    end

    dvs = DestructuredArgs(states(sys), inbounds=!checkbounds)
    ps = DestructuredArgs(parameters(sys), inbounds=!checkbounds)
    ivs = independent_variables(sys)
    args = [dvs, ps, ivs...]

    ex = Func(
        args, [],
        Let(
            map(eq -> eq.lhs←eq.rhs, obs[1:maxidx]),
            isscalar ? output[1] : MakeArray(output, output_type)
           )
    ) |> toexpr

    expression ? ex : @RuntimeGeneratedFunction(ex)
end

function _eq_unordered(a, b)
    length(a) === length(b) || return false
    n = length(a)
    idxs = Set(1:n)
    for x ∈ a
        idx = findfirst(isequal(x), b)
        idx === nothing && return false
        idx ∈ idxs      || return false
        delete!(idxs, idx)
    end
    return true
end

function collect_differential_variables(sys::ODESystem)
    eqs = equations(sys)
    vars = Set()
    diffvars = Set()
    for eq in eqs
        vars!(vars, eq)
        for v in vars
            isdifferential(v) || continue
            push!(diffvars, arguments(v)[1])
        end
        empty!(vars)
    end
    return diffvars
end

# We have a stand-alone function to convert a `NonlinearSystem` or `ODESystem`
# to an `ODESystem` to connect systems, and we later can reply on
# `structural_simplify` to convert `ODESystem`s to `NonlinearSystem`s.
"""
$(TYPEDSIGNATURES)

Convert a `NonlinearSystem` to an `ODESystem` or converts an `ODESystem` to a
new `ODESystem` with a different independent variable.
"""
function convert_system(::Type{<:ODESystem}, sys, t; name=nameof(sys))
    isempty(observed(sys)) || throw(ArgumentError("`convert_system` cannot handle reduced model (i.e. observed(sys) is non-empty)."))
    t = value(t)
    varmap = Dict()
    sts = states(sys)
    newsts = similar(sts, Any)
    for (i, s) in enumerate(sts)
        if istree(s)
            args = arguments(s)
            length(args) == 1 || throw(InvalidSystemException("Illegal state: $s. The state can have at most one argument like `x(t)`."))
            arg = args[1]
            if isequal(arg, t)
                newsts[i] = s
                continue
            end
            ns = operation(s)(t)
            newsts[i] = ns
            varmap[s] = ns
        else
            ns = indepvar2depvar(s, t)
            newsts[i] = ns
            varmap[s] = ns
        end
    end
    sub = Base.Fix2(substitute, varmap)
    neweqs = map(sub, equations(sys))
    defs = Dict(sub(k) => sub(v) for (k, v) in defaults(sys))
    return ODESystem(neweqs, t, newsts, parameters(sys); defaults=defs, name=name)
end
