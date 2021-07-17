function calculate_tgrad(sys::AbstractODESystem;
                         simplify=false)
  isempty(get_tgrad(sys)[]) || return get_tgrad(sys)[]  # use cached tgrad, if possible

  # We need to remove explicit time dependence on the state because when we
  # have `u(t) * t` we want to have the tgrad to be `u(t)` instead of `u'(t) *
  # t + u(t)`.
  rhs = [detime_dvs(eq.rhs) for eq ∈ equations(sys)]
  iv = get_iv(sys)
  xs = states(sys)
  rule = Dict(map((x, xt) -> xt=>x, detime_dvs.(xs), xs))
  rhs = substitute.(rhs, Ref(rule))
  tgrad = [expand_derivatives(ModelingToolkit.Differential(iv)(r), simplify) for r in rhs]
  reverse_rule = Dict(map((x, xt) -> x=>xt, detime_dvs.(xs), xs))
  tgrad = Num.(substitute.(tgrad, Ref(reverse_rule)))
  get_tgrad(sys)[] = tgrad
  return tgrad
end

function calculate_jacobian(sys::AbstractODESystem;
                            sparse=false, simplify=false)
    cache = get_jac(sys)[]
    if cache isa Tuple && cache[2] == (sparse, simplify)
        return cache[1]
    end
    rhs = [eq.rhs for eq ∈ equations(sys)]

    iv = get_iv(sys)
    dvs = states(sys)

    if sparse
        jac = sparsejacobian(rhs, dvs, simplify=simplify)
    else
        jac = jacobian(rhs, dvs, simplify=simplify)
    end

    get_jac(sys)[] = jac, (sparse, simplify) # cache Jacobian
    return jac
end

function calculate_control_jacobian(sys::AbstractODESystem;
                                    sparse=false, simplify=false)
    cache = get_ctrl_jac(sys)[]
    if cache isa Tuple && cache[2] == (sparse, simplify)
        return cache[1]
    end

    rhs = [eq.rhs for eq ∈ equations(sys)]

    iv = get_iv(sys)
    ctrls = controls(sys)

    if sparse
        jac = sparsejacobian(rhs, ctrls, simplify=simplify)
    else
        jac = jacobian(rhs, ctrls, simplify=simplify)
    end

    get_ctrl_jac(sys)[] = jac, (sparse, simplify) # cache Jacobian
    return jac
end

function generate_tgrad(sys::AbstractODESystem, dvs = states(sys), ps = parameters(sys);
                        simplify=false, kwargs...)
    tgrad = calculate_tgrad(sys,simplify=simplify)
    return build_function(tgrad, dvs, ps, get_iv(sys); kwargs...)
end

function generate_jacobian(sys::AbstractODESystem, dvs = states(sys), ps = parameters(sys);
                           simplify=false, sparse = false, kwargs...)
    jac = calculate_jacobian(sys;simplify=simplify,sparse=sparse)
    return build_function(jac, dvs, ps, get_iv(sys); kwargs...)
end

function generate_control_jacobian(sys::AbstractODESystem, dvs = states(sys), ps = parameters(sys);
                                   simplify=false, sparse = false, kwargs...)
    jac = calculate_control_jacobian(sys;simplify=simplify,sparse=sparse)
    return build_function(jac, dvs, ps, get_iv(sys); kwargs...)
end

check_derivative_variables(eq) = check_operator_variables(eq, Differential)

function generate_function(
        sys::AbstractODESystem, dvs = states(sys), ps = parameters(sys);
        implicit_dae=false,
        ddvs=implicit_dae ? map(Differential(get_iv(sys)), dvs) : nothing,
        kwargs...
    )
    # optimization
    #obsvars = map(eq->eq.lhs, observed(sys))
    #fulldvs = [dvs; obsvars]

    eqs = equations(sys)
    foreach(check_derivative_variables, eqs)
    # substitute x(t) by just x
    rhss = implicit_dae ? [_iszero(eq.lhs) ? eq.rhs : eq.rhs - eq.lhs for eq in eqs] :
                          [eq.rhs for eq in eqs]
    #rhss = Let(obss, rhss)

    # TODO: add an optional check on the ordering of observed equations
    u = map(x->time_varying_as_func(value(x), sys), dvs)
    p = map(x->time_varying_as_func(value(x), sys), ps)
    t = get_iv(sys)

    if implicit_dae
        build_function(rhss, ddvs, u, p, t; kwargs...)
    else
        build_function(rhss, u, p, t; kwargs...)
    end
end

function time_varying_as_func(x, sys::AbstractTimeDependentSystem)
    # if something is not x(t) (the current state)
    # but is `x(t-1)` or something like that, pass in `x` as a callable function rather
    # than pass in a value in place of x(t).
    #
    # This is done by just making `x` the argument of the function.
    if istree(x) &&
        operation(x) isa Sym &&
        !(length(arguments(x)) == 1 && isequal(arguments(x)[1], get_iv(sys)))
        return operation(x)
    end
    return x
end

function calculate_massmatrix(sys::AbstractODESystem; simplify=false)
    eqs = equations(sys)
    dvs = states(sys)
    M = zeros(length(eqs),length(eqs))
    state2idx = Dict(s => i for (i, s) in enumerate(dvs))
    for (i,eq) in enumerate(eqs)
        if eq.lhs isa Term && operation(eq.lhs) isa Differential
            st = var_from_nested_derivative(eq.lhs)[1]
            j = state2idx[st]
            M[i,j] = 1
        else
            _iszero(eq.lhs) || error("Only semi-explicit constant mass matrices are currently supported. Faulty equation: $eq.")
        end
    end
    M = simplify ? ModelingToolkit.simplify.(M) : M
    # M should only contain concrete numbers
    M == I ? I : M
end

jacobian_sparsity(sys::AbstractODESystem) =
    jacobian_sparsity([eq.rhs for eq ∈ equations(sys)],
                      [dv for dv in states(sys)])

function isautonomous(sys::AbstractODESystem)
    tgrad = calculate_tgrad(sys;simplify=true)
    all(iszero,tgrad)
end

for F in [:ODEFunction, :DAEFunction]
    @eval function DiffEqBase.$F(sys::AbstractODESystem, args...; kwargs...)
        $F{true}(sys, args...; kwargs...)
    end
end

"""
```julia
function DiffEqBase.ODEFunction{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys);
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     sparse = false,
                                     kwargs...) where {iip}
```

Create an `ODEFunction` from the [`ODESystem`](@ref). The arguments `dvs` and `ps`
are used to set the order of the dependent variable and parameter vectors,
respectively.
"""
function DiffEqBase.ODEFunction{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys), u0 = nothing;
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     eval_expression = true,
                                     sparse = false, simplify=false,
                                     eval_module = @__MODULE__,
                                     steady_state = false,
                                     checkbounds=false,
                                     kwargs...) where {iip}

    f_gen = generate_function(sys, dvs, ps; expression=Val{eval_expression}, expression_module=eval_module, checkbounds=checkbounds, kwargs...)
    f_oop,f_iip = eval_expression ? (@RuntimeGeneratedFunction(eval_module, ex) for ex in f_gen) : f_gen
    f(u,p,t) = f_oop(u,p,t)
    f(du,u,p,t) = f_iip(du,u,p,t)

    if tgrad
        tgrad_gen = generate_tgrad(sys, dvs, ps;
                                   simplify=simplify,
                                   expression=Val{eval_expression}, expression_module=eval_module,
                                   checkbounds=checkbounds, kwargs...)
        tgrad_oop,tgrad_iip = eval_expression ? (@RuntimeGeneratedFunction(eval_module, ex) for ex in tgrad_gen) : tgrad_gen
        _tgrad(u,p,t) = tgrad_oop(u,p,t)
        _tgrad(J,u,p,t) = tgrad_iip(J,u,p,t)
    else
        _tgrad = nothing
    end

    if jac
        jac_gen = generate_jacobian(sys, dvs, ps;
                                    simplify=simplify, sparse = sparse,
                                    expression=Val{eval_expression}, expression_module=eval_module,
                                    checkbounds=checkbounds, kwargs...)
        jac_oop,jac_iip = eval_expression ? (@RuntimeGeneratedFunction(eval_module, ex) for ex in jac_gen) : jac_gen
        _jac(u,p,t) = jac_oop(u,p,t)
        _jac(J,u,p,t) = jac_iip(J,u,p,t)
    else
        _jac = nothing
    end

    M = calculate_massmatrix(sys)

    _M = (u0 === nothing || M == I) ? M : ArrayInterface.restructure(u0 .* u0',M)

    obs = observed(sys)
    observedfun = if steady_state
        isempty(obs) ? SciMLBase.DEFAULT_OBSERVED_NO_TIME : let sys = sys, dict = Dict()
            function generated_observed(obsvar, u, p, t=Inf)
                obs = get!(dict, value(obsvar)) do
                    build_explicit_observed_function(sys, obsvar)
                end
                obs(u, p, t)
            end
        end
    else
        isempty(obs) ? SciMLBase.DEFAULT_OBSERVED : let sys = sys, dict = Dict()
            function generated_observed(obsvar, u, p, t)
                obs = get!(dict, value(obsvar)) do
                    build_explicit_observed_function(sys, obsvar; checkbounds=checkbounds)
                end
                obs(u, p, t)
            end
        end
    end

    jac_prototype = if sparse
        uElType = u0 === nothing ? Float64 : eltype(u0)
        if jac
            similar(calculate_jacobian(sys, sparse=sparse), uElType)
        else
            similar(jacobian_sparsity(sys), uElType)
        end
    else
        nothing
    end
    ODEFunction{iip}(
                     f,
                     jac = _jac === nothing ? nothing : _jac,
                     tgrad = _tgrad === nothing ? nothing : _tgrad,
                     mass_matrix = _M,
                     jac_prototype = jac_prototype,
                     syms = Symbol.(states(sys)),
                     indepsym = Symbol(get_iv(sys)),
                     observed = observedfun,
                    )
end

"""
```julia
function DiffEqBase.DAEFunction{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys);
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     sparse = false,
                                     kwargs...) where {iip}
```

Create an `DAEFunction` from the [`ODESystem`](@ref). The arguments `dvs` and
`ps` are used to set the order of the dependent variable and parameter vectors,
respectively.
"""
function DiffEqBase.DAEFunction{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys), u0 = nothing;
                                     ddvs=map(diff2term ∘ Differential(get_iv(sys)), dvs),
                                     version = nothing,
                                     #=
                                     tgrad=false,
                                     jac = false,
                                     sparse = false,
                                     =#
                                     simplify=false,
                                     eval_expression = true,
                                     eval_module = @__MODULE__,
                                     kwargs...) where {iip}

    f_gen = generate_function(sys, dvs, ps; implicit_dae = true, expression=Val{eval_expression}, expression_module=eval_module, kwargs...)
    f_oop,f_iip = eval_expression ? (@RuntimeGeneratedFunction(eval_module, ex) for ex in f_gen) : f_gen
    f(du,u,p,t) = f_oop(du,u,p,t)
    f(out,du,u,p,t) = f_iip(out,du,u,p,t)

    # TODO: Jacobian sparsity / sparse Jacobian / dense Jacobian

    #=
    observedfun = let sys = sys, dict = Dict()
        # TODO: We don't have enought information to reconstruct arbitrary state
        # in general from `(u, p, t)`, e.g. `a ~ D(x)`.
        function generated_observed(obsvar, u, p, t)
            obs = get!(dict, value(obsvar)) do
                build_explicit_observed_function(sys, obsvar)
            end
            obs(u, p, t)
        end
    end
    =#

    DAEFunction{iip}(
                     f,
                     syms = Symbol.(dvs),
                     # missing fields in `DAEFunction`
                     #indepsym = Symbol(get_iv(sys)),
                     #observed = observedfun,
                    )
end

"""
```julia
function ODEFunctionExpr{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys);
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     sparse = false,
                                     kwargs...) where {iip}
```

Create a Julia expression for an `ODEFunction` from the [`ODESystem`](@ref).
The arguments `dvs` and `ps` are used to set the order of the dependent
variable and parameter vectors, respectively.
"""
struct ODEFunctionExpr{iip} end

struct ODEFunctionClosure{O, I} <: Function
    f_oop::O
    f_iip::I
end
(f::ODEFunctionClosure)(u, p, t) = f.f_oop(u, p, t)
(f::ODEFunctionClosure)(du, u, p, t) = f.f_iip(du, u, p, t)

function ODEFunctionExpr{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys), u0 = nothing;
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     linenumbers = false,
                                     sparse = false, simplify=false,
                                     steady_state = false,
                                     kwargs...) where {iip}

    f_oop, f_iip = generate_function(sys, dvs, ps; expression=Val{true}, kwargs...)

    dict = Dict()
    #=
    observedfun = if steady_state
        :(function generated_observed(obsvar, u, p, t=Inf)
              obs = get!($dict, value(obsvar)) do
                  build_explicit_observed_function($sys, obsvar)
              end
              obs(u, p, t)
          end)
    else
        :(function generated_observed(obsvar, u, p, t)
              obs = get!($dict, value(obsvar)) do
                  build_explicit_observed_function($sys, obsvar)
              end
              obs(u, p, t)
          end)
    end
    =#

    fsym = gensym(:f)
    _f = :($fsym = ModelingToolkit.ODEFunctionClosure($f_oop, $f_iip))
    tgradsym = gensym(:tgrad)
    if tgrad
        tgrad_oop, tgrad_iip = generate_tgrad(sys, dvs, ps;
                                simplify=simplify,
                                expression=Val{true}, kwargs...)
        _tgrad = :($tgradsym = ModelingToolkit.ODEFunctionClosure($tgrad_oop, $tgrad_iip))
    else
        _tgrad = :($tgradsym = nothing)
    end

    jacsym = gensym(:jac)
    if jac
        jac_oop,jac_iip = generate_jacobian(sys, dvs, ps;
                                 sparse=sparse, simplify=simplify,
                                 expression=Val{true}, kwargs...)
        _jac = :($jacsym = ModelingToolkit.ODEFunctionClosure($jac_oop, $jac_iip))
    else
        _jac = :($jacsym = nothing)
    end

    M = calculate_massmatrix(sys)

    _M = (u0 === nothing || M == I) ? M : ArrayInterface.restructure(u0 .* u0',M)

    jp_expr = sparse ? :(similar($(get_jac(sys)[]),Float64)) : :nothing
    ex = quote
        $_f
        $_tgrad
        $_jac
        M = $_M
        ODEFunction{$iip}(
                          $fsym,
                          jac = $jacsym,
                          tgrad = $tgradsym,
                          mass_matrix = M,
                          jac_prototype = $jp_expr,
                          syms = $(Symbol.(states(sys))),
                          indepsym = $(QuoteNode(Symbol(get_iv(sys)))),
                         )
    end
    !linenumbers ? striplines(ex) : ex
end

function process_DEProblem(constructor, sys::AbstractODESystem,u0map,parammap;
                           implicit_dae = false, du0map = nothing,
                           version = nothing, tgrad=false,
                           jac = false,
                           checkbounds = false, sparse = false,
                           simplify=false,
                           linenumbers = true, parallel=SerialForm(),
                           eval_expression = true,
                           kwargs...)
    eqs = equations(sys)
    dvs = states(sys)
    ps = parameters(sys)
    defs = defaults(sys)
    iv = get_iv(sys)
    if parammap isa Dict
        u0defs = merge(parammap, defs)
    elseif eltype(parammap) <: Pair
        u0defs = merge(Dict(parammap), defs)
    elseif eltype(parammap) <: Number
        u0defs = merge(Dict(zip(ps, parammap)), defs)
    else
        u0defs = defs
    end
    if u0map isa Dict
        pdefs = merge(u0map, defs)
    elseif eltype(u0map) <: Pair
        pdefs = merge(Dict(u0map), defs)
    elseif eltype(u0map) <: Number
        pdefs = merge(Dict(zip(dvs, u0map)), defs)
    else
        pdefs = defs
    end

    u0 = varmap_to_vars(u0map,dvs; defaults=u0defs)
    if implicit_dae && du0map !== nothing
        ddvs = map(Differential(iv), dvs)
        du0 = varmap_to_vars(du0map, ddvs; defaults=defaults, toterm=identity)
    else
        du0 = nothing
        ddvs = nothing
    end
    p = varmap_to_vars(parammap,ps; defaults=pdefs)

    check_eqs_u0(eqs, dvs, u0; kwargs...)

    f = constructor(sys,dvs,ps,u0;ddvs=ddvs,tgrad=tgrad,jac=jac,checkbounds=checkbounds,
                    linenumbers=linenumbers,parallel=parallel,simplify=simplify,
                    sparse=sparse,eval_expression=eval_expression,kwargs...)
    implicit_dae ? (f, du0, u0, p) : (f, u0, p)
end

function ODEFunctionExpr(sys::AbstractODESystem, args...; kwargs...)
    ODEFunctionExpr{true}(sys, args...; kwargs...)
end

"""
```julia
function DAEFunctionExpr{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys);
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     sparse = false,
                                     kwargs...) where {iip}
```

Create a Julia expression for an `ODEFunction` from the [`ODESystem`](@ref).
The arguments `dvs` and `ps` are used to set the order of the dependent
variable and parameter vectors, respectively.
"""
struct DAEFunctionExpr{iip} end

struct DAEFunctionClosure{O, I} <: Function
    f_oop::O
    f_iip::I
end
(f::DAEFunctionClosure)(du, u, p, t) = f.f_oop(du, u, p, t)
(f::DAEFunctionClosure)(out, du, u, p, t) = f.f_iip(out, du, u, p, t)

function DAEFunctionExpr{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys), u0 = nothing;
                                     version = nothing, tgrad=false,
                                     jac = false,
                                     linenumbers = false,
                                     sparse = false, simplify=false,
                                     kwargs...) where {iip}
    f_oop, f_iip = generate_function(sys, dvs, ps; expression=Val{true}, implicit_dae = true, kwargs...)
    fsym = gensym(:f)
    _f = :($fsym = $DAEFunctionClosure($f_oop, $f_iip))
    ex = quote
        $_f
        ODEFunction{$iip}($fsym,)
    end
    !linenumbers ? striplines(ex) : ex
end

function DAEFunctionExpr(sys::AbstractODESystem, args...; kwargs...)
    DAEFunctionExpr{true}(sys, args...; kwargs...)
end

for P in [:ODEProblem, :DAEProblem]
    @eval function DiffEqBase.$P(sys::AbstractODESystem, args...; kwargs...)
        $P{true}(sys, args...; kwargs...)
    end
end

"""
```julia
function DiffEqBase.ODEProblem{iip}(sys::AbstractODESystem,u0map,tspan,
                                    parammap=DiffEqBase.NullParameters();
                                    version = nothing, tgrad=false,
                                    jac = false,
                                    checkbounds = false, sparse = false,
                                    simplify=false,
                                    linenumbers = true, parallel=SerialForm(),
                                    kwargs...) where iip
```

Generates an ODEProblem from an ODESystem and allows for automatically
symbolically calculating numerical enhancements.
"""
function DiffEqBase.ODEProblem{iip}(sys::AbstractODESystem,u0map,tspan,
                                    parammap=DiffEqBase.NullParameters();kwargs...) where iip
    f, u0, p = process_DEProblem(ODEFunction{iip}, sys, u0map, parammap; kwargs...)
    ODEProblem{iip}(f,u0,tspan,p;kwargs...)
end

"""
```julia
function DiffEqBase.DAEProblem{iip}(sys::AbstractODESystem,u0map,tspan,
                                    parammap=DiffEqBase.NullParameters();
                                    version = nothing, tgrad=false,
                                    jac = false,
                                    checkbounds = false, sparse = false,
                                    simplify=false,
                                    linenumbers = true, parallel=SerialForm(),
                                    kwargs...) where iip
```

Generates an DAEProblem from an ODESystem and allows for automatically
symbolically calculating numerical enhancements.
"""
function DiffEqBase.DAEProblem{iip}(sys::AbstractODESystem,du0map,u0map,tspan,
                                    parammap=DiffEqBase.NullParameters();kwargs...) where iip
    f, du0, u0, p = process_DEProblem(
        DAEFunction{iip}, sys, u0map, parammap;
        implicit_dae=true, du0map=du0map, kwargs...
    )
    diffvars = collect_differential_variables(sys)
    sts = states(sys)
    differential_vars = map(Base.Fix2(in, diffvars), sts)
    DAEProblem{iip}(f,du0,u0,tspan,p;differential_vars=differential_vars,kwargs...)
end

"""
```julia
function ODEProblemExpr{iip}(sys::AbstractODESystem,u0map,tspan,
                                    parammap=DiffEqBase.NullParameters();
                                    version = nothing, tgrad=false,
                                    jac = false,
                                    checkbounds = false, sparse = false,
                                    linenumbers = true, parallel=SerialForm(),
                                    skipzeros=true, fillzeros=true,
                                    simplify=false,
                                    kwargs...) where iip
```

Generates a Julia expression for constructing an ODEProblem from an
ODESystem and allows for automatically symbolically calculating
numerical enhancements.
"""
struct ODEProblemExpr{iip} end

function ODEProblemExpr{iip}(sys::AbstractODESystem,u0map,tspan,
                             parammap=DiffEqBase.NullParameters();
                             kwargs...) where iip

    f, u0, p = process_DEProblem(ODEFunctionExpr{iip}, sys, u0map, parammap; kwargs...)
    linenumbers = get(kwargs, :linenumbers, true)

    ex = quote
        f = $f
        u0 = $u0
        tspan = $tspan
        p = $p
        ODEProblem(f,u0,tspan,p;$(kwargs...))
    end
    !linenumbers ? striplines(ex) : ex
end

function ODEProblemExpr(sys::AbstractODESystem, args...; kwargs...)
    ODEProblemExpr{true}(sys, args...; kwargs...)
end

"""
```julia
function DAEProblemExpr{iip}(sys::AbstractODESystem,u0map,tspan,
                                    parammap=DiffEqBase.NullParameters();
                                    version = nothing, tgrad=false,
                                    jac = false,
                                    checkbounds = false, sparse = false,
                                    linenumbers = true, parallel=SerialForm(),
                                    skipzeros=true, fillzeros=true,
                                    simplify=false,
                                    kwargs...) where iip
```

Generates a Julia expression for constructing an ODEProblem from an
ODESystem and allows for automatically symbolically calculating
numerical enhancements.
"""
struct DAEProblemExpr{iip} end

function DAEProblemExpr{iip}(sys::AbstractODESystem,du0map,u0map,tspan,
                             parammap=DiffEqBase.NullParameters();
                             kwargs...) where iip
    f, du0, u0, p = process_DEProblem(
        DAEFunctionExpr{iip}, sys, u0map, parammap;
        implicit_dae=true, du0map=du0map, kwargs...
    )
    linenumbers = get(kwargs, :linenumbers, true)
    diffvars = collect_differential_variables(sys)
    sts = states(sys)
    differential_vars = map(Base.Fix2(in, diffvars), sts)

    ex = quote
        f = $f
        u0 = $u0
        du0 = $du0
        tspan = $tspan
        p = $p
        differential_vars = $differential_vars
        DAEProblem{$iip}(f,du0,u0,tspan,p;differential_vars=differential_vars,$(kwargs...))
    end
    !linenumbers ? striplines(ex) : ex
end

function DAEProblemExpr(sys::AbstractODESystem, args...; kwargs...)
    DAEProblemExpr{true}(sys, args...; kwargs...)
end


### Enables Steady State Problems ###
function DiffEqBase.SteadyStateProblem(sys::AbstractODESystem, args...; kwargs...)
    SteadyStateProblem{true}(sys, args...; kwargs...)
end

"""
```julia
function DiffEqBase.SteadyStateProblem(sys::AbstractODESystem,u0map,
                                    parammap=DiffEqBase.NullParameters();
                                    version = nothing, tgrad=false,
                                    jac = false,
                                    checkbounds = false, sparse = false,
                                    linenumbers = true, parallel=SerialForm(),
                                    kwargs...) where iip
```
Generates an SteadyStateProblem from an ODESystem and allows for automatically
symbolically calculating numerical enhancements.
"""
function DiffEqBase.SteadyStateProblem{iip}(sys::AbstractODESystem,u0map,
                                            parammap=DiffEqBase.NullParameters();
                                            kwargs...) where iip
    f, u0, p = process_DEProblem(ODEFunction{iip}, sys, u0map, parammap; steady_state = true, kwargs...)
    SteadyStateProblem{iip}(f,u0,p;kwargs...)
end

"""
```julia
function DiffEqBase.SteadyStateProblemExpr(sys::AbstractODESystem,u0map,
                                    parammap=DiffEqBase.NullParameters();
                                    version = nothing, tgrad=false,
                                    jac = false,
                                    checkbounds = false, sparse = false,
                                    skipzeros=true, fillzeros=true,
                                    linenumbers = true, parallel=SerialForm(),
                                    kwargs...) where iip
```
Generates a Julia expression for building a SteadyStateProblem from
an ODESystem and allows for automatically symbolically calculating
numerical enhancements.
"""
struct SteadyStateProblemExpr{iip} end

function SteadyStateProblemExpr{iip}(sys::AbstractODESystem,u0map,
                                    parammap=DiffEqBase.NullParameters();
                                    kwargs...) where iip
    f, u0, p = process_DEProblem(ODEFunctionExpr{iip}, sys, u0map, parammap;steady_state = true, kwargs...)
    linenumbers = get(kwargs, :linenumbers, true)
    ex = quote
        f = $f
        u0 = $u0
        p = $p
        SteadyStateProblem(f,u0,p;$(kwargs...))
    end
    !linenumbers ? striplines(ex) : ex
end

function SteadyStateProblemExpr(sys::AbstractODESystem, args...; kwargs...)
    SteadyStateProblemExpr{true}(sys, args...; kwargs...)
end

isdifferential(expr) = istree(expr) && operation(expr) isa Differential
isdiffeq(eq) = isdifferential(eq.lhs)
