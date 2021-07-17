"""
$(TYPEDEF)

A system of stochastic differential equations.

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

noiseeqs = [0.1*x,
            0.1*y,
            0.1*z]

de = SDESystem(eqs,noiseeqs,t,[x,y,z],[σ,ρ,β])
```
"""
struct SDESystem <: AbstractODESystem
    """The expressions defining the drift term."""
    eqs::Vector{Equation}
    """The expressions defining the diffusion term."""
    noiseeqs::AbstractArray
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
    tgrad::RefValue
    """
    Jacobian matrix. Note: this field will not be defined until
    [`calculate_jacobian`](@ref) is called on the system.
    """
    jac::RefValue
    """
    Control Jacobian matrix. Note: this field will not be defined until
    [`calculate_control_jacobian`](@ref) is called on the system.
    """
    ctrl_jac::RefValue{Any}
    """
    `Wfact` matrix. Note: this field will not be defined until
    [`generate_factorized_W`](@ref) is called on the system.
    """
    Wfact::RefValue
    """
    `Wfact_t` matrix. Note: this field will not be defined until
    [`generate_factorized_W`](@ref) is called on the system.
    """
    Wfact_t::RefValue
    """
    Name: the name of the system
    """
    name::Symbol
    """
    Systems: the internal systems. These are required to have unique names.
    """
    systems::Vector{SDESystem}
    """
    defaults: The default values to use when initial conditions and/or
    parameters are not supplied in `ODEProblem`.
    """
    defaults::Dict
    """
    type: type of the system
    """
    connection_type::Any

    function SDESystem(deqs, neqs, iv, dvs, ps, ctrls, observed, tgrad, jac, ctrl_jac, Wfact, Wfact_t, name, systems, defaults, connection_type)
        check_variables(dvs,iv)
        check_parameters(ps,iv)
        check_equations(deqs,iv)
        new(deqs, neqs, iv, dvs, ps, ctrls, observed, tgrad, jac, ctrl_jac, Wfact, Wfact_t, name, systems, defaults, connection_type)
    end
end

function SDESystem(deqs::AbstractVector{<:Equation}, neqs, iv, dvs, ps;
                   controls = Num[],
                   observed = Num[],
                   systems = SDESystem[],
                   default_u0=Dict(),
                   default_p=Dict(),
                   defaults=_merge(Dict(default_u0), Dict(default_p)),
                   name = gensym(:SDESystem),
                   connection_type=nothing,
                   )
    deqs = collect(deqs)
    iv′ = value(iv)
    dvs′ = value.(dvs)
    ps′ = value.(ps)
    ctrl′ = value.(controls)

    sysnames = nameof.(systems)
    if length(unique(sysnames)) != length(sysnames)
        throw(ArgumentError("System names must be unique."))
    end
    if !(isempty(default_u0) && isempty(default_p))
        Base.depwarn("`default_u0` and `default_p` are deprecated. Use `defaults` instead.", :SDESystem, force=true)
    end
    defaults = todict(defaults)
    defaults = Dict(value(k) => value(v) for (k, v) in pairs(defaults))

    collect_defaults!(defaults, dvs′)
    collect_defaults!(defaults, ps′)

    tgrad = RefValue(Vector{Num}(undef, 0))
    jac = RefValue{Any}(Matrix{Num}(undef, 0, 0))
    ctrl_jac = RefValue{Any}(Matrix{Num}(undef, 0, 0))
    Wfact   = RefValue(Matrix{Num}(undef, 0, 0))
    Wfact_t = RefValue(Matrix{Num}(undef, 0, 0))
    SDESystem(deqs, neqs, iv′, dvs′, ps′, ctrl′, observed, tgrad, jac, ctrl_jac, Wfact, Wfact_t, name, systems, defaults, connection_type)
end

function generate_diffusion_function(sys::SDESystem, dvs = states(sys), ps = parameters(sys); kwargs...)
    return build_function(get_noiseeqs(sys),
                          map(x->time_varying_as_func(value(x), sys), dvs),
                          map(x->time_varying_as_func(value(x), sys), ps),
                          get_iv(sys); kwargs...)
end

"""
$(TYPEDSIGNATURES)

Choose correction_factor=-1//2 (1//2) to converte Ito -> Stratonovich (Stratonovich->Ito).
"""
function stochastic_integral_transform(sys::SDESystem, correction_factor)
    # use the general interface
    if typeof(get_noiseeqs(sys)) <: Vector
        eqs = vcat([equations(sys)[i].lhs ~ get_noiseeqs(sys)[i] for i in eachindex(states(sys))]...)
        de = ODESystem(eqs,get_iv(sys),states(sys),parameters(sys))

        jac = calculate_jacobian(de, sparse=false, simplify=false)
        ∇σσ′ = simplify.(jac*get_noiseeqs(sys))

        deqs = vcat([equations(sys)[i].lhs ~ equations(sys)[i].rhs+ correction_factor*∇σσ′[i] for i in eachindex(states(sys))]...)
    else
        dimstate, m = size(get_noiseeqs(sys))
        eqs = vcat([equations(sys)[i].lhs ~ get_noiseeqs(sys)[i] for i in eachindex(states(sys))]...)
        de = ODESystem(eqs,get_iv(sys),states(sys),parameters(sys))

        jac = calculate_jacobian(de, sparse=false, simplify=false)
        ∇σσ′ = simplify.(jac*get_noiseeqs(sys)[:,1])
        for k = 2:m
            eqs = vcat([equations(sys)[i].lhs ~ get_noiseeqs(sys)[Int(i+(k-1)*dimstate)] for i in eachindex(states(sys))]...)
            de = ODESystem(eqs,get_iv(sys),states(sys),parameters(sys))

            jac = calculate_jacobian(de, sparse=false, simplify=false)
            ∇σσ′ = ∇σσ′ + simplify.(jac*get_noiseeqs(sys)[:,k])
        end

        deqs = vcat([equations(sys)[i].lhs ~ equations(sys)[i].rhs + correction_factor*∇σσ′[i] for i in eachindex(states(sys))]...)
    end


    SDESystem(deqs,get_noiseeqs(sys),get_iv(sys),states(sys),parameters(sys))
end

"""
```julia
function DiffEqBase.SDEFunction{iip}(sys::SDESystem, dvs = sys.states, ps = sys.ps;
                                     version = nothing, tgrad=false, sparse = false,
                                     jac = false, Wfact = false, kwargs...) where {iip}
```

Create an `SDEFunction` from the [`SDESystem`](@ref). The arguments `dvs` and `ps`
are used to set the order of the dependent variable and parameter vectors,
respectively.
"""
function DiffEqBase.SDEFunction{iip}(sys::SDESystem, dvs = states(sys), ps = parameters(sys),
                                     u0 = nothing;
                                     version = nothing, tgrad=false, sparse = false,
                                     jac = false, Wfact = false, eval_expression = true, kwargs...) where {iip}
    dvs = scalarize.(dvs)
    ps = scalarize.(ps)

    f_gen = generate_function(sys, dvs, ps; expression=Val{eval_expression}, kwargs...)
    f_oop,f_iip = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in f_gen) : f_gen
    g_gen = generate_diffusion_function(sys, dvs, ps; expression=Val{eval_expression}, kwargs...)
    g_oop,g_iip = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in g_gen) : g_gen

    f(u,p,t) = f_oop(u,p,t)
    f(du,u,p,t) = f_iip(du,u,p,t)
    g(u,p,t) = g_oop(u,p,t)
    g(du,u,p,t) = g_iip(du,u,p,t)

    if tgrad
        tgrad_gen = generate_tgrad(sys, dvs, ps; expression=Val{eval_expression}, kwargs...)
        tgrad_oop,tgrad_iip = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in tgrad_gen) : tgrad_gen
        _tgrad(u,p,t) = tgrad_oop(u,p,t)
        _tgrad(J,u,p,t) = tgrad_iip(J,u,p,t)
    else
        _tgrad = nothing
    end

    if jac
        jac_gen = generate_jacobian(sys, dvs, ps; expression=Val{eval_expression}, sparse=sparse, kwargs...)
        jac_oop,jac_iip = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in jac_gen) : jac_gen
        _jac(u,p,t) = jac_oop(u,p,t)
        _jac(J,u,p,t) = jac_iip(J,u,p,t)
    else
        _jac = nothing
    end

    if Wfact
        tmp_Wfact,tmp_Wfact_t = generate_factorized_W(sys, dvs, ps, true; expression=Val{true}, kwargs...)
        Wfact_oop, Wfact_iip = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in tmp_Wfact) : tmp_Wfact
        Wfact_oop_t, Wfact_iip_t = eval_expression ? (@RuntimeGeneratedFunction(ex) for ex in tmp_Wfact_t) : tmp_Wfact_t
        _Wfact(u,p,dtgamma,t) = Wfact_oop(u,p,dtgamma,t)
        _Wfact(W,u,p,dtgamma,t) = Wfact_iip(W,u,p,dtgamma,t)
        _Wfact_t(u,p,dtgamma,t) = Wfact_oop_t(u,p,dtgamma,t)
        _Wfact_t(W,u,p,dtgamma,t) = Wfact_iip_t(W,u,p,dtgamma,t)
    else
        _Wfact,_Wfact_t = nothing,nothing
    end

    M = calculate_massmatrix(sys)
    _M = (u0 === nothing || M == I) ? M : ArrayInterface.restructure(u0 .* u0',M)

    sts = states(sys)
    SDEFunction{iip}(f,g,
                     jac = _jac === nothing ? nothing : _jac,
                     tgrad = _tgrad === nothing ? nothing : _tgrad,
                     Wfact = _Wfact === nothing ? nothing : _Wfact,
                     Wfact_t = _Wfact_t === nothing ? nothing : _Wfact_t,
                     mass_matrix = _M,
                     syms = Symbol.(states(sys)))
end

function DiffEqBase.SDEFunction(sys::SDESystem, args...; kwargs...)
    SDEFunction{true}(sys, args...; kwargs...)
end

"""
```julia
function DiffEqBase.SDEFunctionExpr{iip}(sys::AbstractODESystem, dvs = states(sys),
                                     ps = parameters(sys);
                                     version = nothing, tgrad=false,
                                     jac = false, Wfact = false,
                                     skipzeros = true, fillzeros = true,
                                     sparse = false,
                                     kwargs...) where {iip}
```

Create a Julia expression for an `SDEFunction` from the [`SDESystem`](@ref).
The arguments `dvs` and `ps` are used to set the order of the dependent
variable and parameter vectors, respectively.
"""
struct SDEFunctionExpr{iip} end

function SDEFunctionExpr{iip}(sys::SDESystem, dvs = states(sys),
                                     ps = parameters(sys), u0 = nothing;
                                     version = nothing, tgrad=false,
                                     jac = false, Wfact = false,
                                     sparse = false,linenumbers = false,
                                     kwargs...) where {iip}

    idx = iip ? 2 : 1
    f = generate_function(sys, dvs, ps; expression=Val{true}, kwargs...)[idx]
    g = generate_diffusion_function(sys, dvs, ps; expression=Val{true}, kwargs...)[idx]
    if tgrad
        _tgrad = generate_tgrad(sys, dvs, ps; expression=Val{true}, kwargs...)[idx]
    else
        _tgrad = :nothing
    end

    if jac
        _jac = generate_jacobian(sys, dvs, ps; sparse = sparse, expression=Val{true}, kwargs...)[idx]
    else
        _jac = :nothing
    end

    if Wfact
        tmp_Wfact,tmp_Wfact_t = generate_factorized_W(sys, dvs, ps; expression=Val{true}, kwargs...)
        _Wfact =  tmp_Wfact[idx]
        _Wfact_t =  tmp_Wfact_t[idx]
    else
        _Wfact,_Wfact_t = :nothing,:nothing
    end

    M = calculate_massmatrix(sys)

    _M = (u0 === nothing || M == I) ? M : ArrayInterface.restructure(u0 .* u0',M)

    ex = quote
        f = $f
        g = $g
        tgrad = $_tgrad
        jac = $_jac
        Wfact = $_Wfact
        Wfact_t = $_Wfact_t
        M = $_M
        SDEFunction{$iip}(f,g,
                         jac = jac,
                         tgrad = tgrad,
                         Wfact = Wfact,
                         Wfact_t = Wfact_t,
                         mass_matrix = M,
                         syms = $(Symbol.(states(sys))))
    end
    !linenumbers ? striplines(ex) : ex
end


function SDEFunctionExpr(sys::SDESystem, args...; kwargs...)
    SDEFunctionExpr{true}(sys, args...; kwargs...)
end

function rename(sys::SDESystem,name)
    SDESystem(sys.eqs, sys.noiseeqs, sys.iv, sys.states, sys.ps, sys.tgrad, sys.jac, sys.Wfact, sys.Wfact_t, name, sys.systems)
end

"""
```julia
function DiffEqBase.SDEProblem{iip}(sys::SDESystem,u0map,tspan,p=parammap;
                                    version = nothing, tgrad=false,
                                    jac = false, Wfact = false,
                                    checkbounds = false, sparse = false,
                                    sparsenoise = sparse,
                                    skipzeros = true, fillzeros = true,
                                    linenumbers = true, parallel=SerialForm(),
                                    kwargs...)
```

Generates an SDEProblem from an SDESystem and allows for automatically
symbolically calculating numerical enhancements.
"""
function DiffEqBase.SDEProblem{iip}(sys::SDESystem,u0map,tspan,parammap=DiffEqBase.NullParameters();
                                    sparsenoise = nothing,
                                    kwargs...) where iip
    f, u0, p = process_DEProblem(SDEFunction{iip}, sys, u0map, parammap; kwargs...)
    sparsenoise === nothing && (sparsenoise = get(kwargs, :sparse, false))

    noiseeqs = get_noiseeqs(sys)
    if noiseeqs isa AbstractVector
        noise_rate_prototype = nothing
    elseif sparsenoise
        I,J,V = findnz(SparseArrays.sparse(noiseeqs))
        noise_rate_prototype = SparseArrays.sparse(I,J,zero(eltype(u0)))
    else
        noise_rate_prototype = zeros(eltype(u0),size(noiseeqs))
    end

    SDEProblem{iip}(f,f.g,u0,tspan,p;noise_rate_prototype=noise_rate_prototype,kwargs...)
end

function DiffEqBase.SDEProblem(sys::SDESystem, args...; kwargs...)
    SDEProblem{true}(sys, args...; kwargs...)
end

"""
```julia
function DiffEqBase.SDEProblemExpr{iip}(sys::AbstractODESystem,u0map,tspan,
                                    parammap=DiffEqBase.NullParameters();
                                    version = nothing, tgrad=false,
                                    jac = false, Wfact = false,
                                    checkbounds = false, sparse = false,
                                    linenumbers = true, parallel=SerialForm(),
                                    kwargs...) where iip
```

Generates a Julia expression for constructing an ODEProblem from an
ODESystem and allows for automatically symbolically calculating
numerical enhancements.
"""
struct SDEProblemExpr{iip} end

function SDEProblemExpr{iip}(sys::SDESystem,u0map,tspan,
                             parammap=DiffEqBase.NullParameters();
                             sparsenoise = nothing,
                             kwargs...) where iip
    f, u0, p = process_DEProblem(SDEFunctionExpr{iip}, sys, u0map, parammap; kwargs...)
    linenumbers = get(kwargs, :linenumbers, true)
    sparsenoise === nothing && (sparsenoise = get(kwargs, :sparse, false))

    noiseeqs = get_noiseeqs(sys)
    if noiseeqs isa AbstractVector
        noise_rate_prototype = nothing
    elseif sparsenoise
        I,J,V = findnz(SparseArrays.sparse(noiseeqs))
        noise_rate_prototype = SparseArrays.sparse(I,J,zero(eltype(u0)))
    else
        T = u0 === nothing ? Float64 : eltype(u0)
        noise_rate_prototype = zeros(T,size(get_noiseeqs(sys)))
    end
    ex = quote
        f = $f
        u0 = $u0
        tspan = $tspan
        p = $p
        noise_rate_prototype = $noise_rate_prototype
        SDEProblem(f,f.g,u0,tspan,p;noise_rate_prototype=noise_rate_prototype,$(kwargs...))
    end
    !linenumbers ? striplines(ex) : ex
end

function SDEProblemExpr(sys::SDESystem, args...; kwargs...)
    SDEProblemExpr{true}(sys, args...; kwargs...)
end
