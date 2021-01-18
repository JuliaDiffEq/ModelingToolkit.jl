"""
$(TYPEDSIGNATURES)

Generate `ODESystem`, dependent variables, and parameters from an `ODEProblem`.
"""
function modelingtoolkitize(prob::DiffEqBase.ODEProblem)
    prob.f isa DiffEqBase.AbstractParameterizedFunction &&
                            return (prob.f.sys, prob.f.sys.states, prob.f.sys.ps)
    @parameters t

    if prob.p isa Tuple || prob.p isa NamedTuple
        p = [x for x in prob.p]
    else
        p = prob.p
    end

    var(x, i) = Num(Sym{FnType{Tuple{symtype(t)}, Real}}(nameof(Variable(x, i))))
    vars = reshape([var(:x, i)(value(t)) for i in eachindex(prob.u0)],size(prob.u0))
    params = p isa DiffEqBase.NullParameters ? [] :
             reshape([Num(Sym{Real}(nameof(Variable(:α, i)))) for i in eachindex(p)],size(p))

    D = Differential(t)

    rhs = [D(var) for var in vars]

    if DiffEqBase.isinplace(prob)
        lhs = similar(vars, Num)
        prob.f(lhs, vars, params, t)
    else
        lhs = prob.f(vars, params, t)
    end

    eqs = vcat([rhs[i] ~ lhs[i] for i in eachindex(prob.u0)]...)
    de = ODESystem(eqs,t,vec(vars),vec(params))

    de
end



"""
$(TYPEDSIGNATURES)

Generate `SDESystem`, dependent variables, and parameters from an `SDEProblem`.
"""
function modelingtoolkitize(prob::DiffEqBase.SDEProblem)
    prob.f isa DiffEqBase.AbstractParameterizedFunction &&
                            return (prob.f.sys, prob.f.sys.states, prob.f.sys.ps)
    @parameters t
    if prob.p isa Tuple || prob.p isa NamedTuple
        p = [x for x in prob.p]
    else
        p = prob.p
    end
    var(x, i) = Num(Sym{FnType{Tuple{symtype(t)}, Real}}(nameof(Variable(x, i))))
    vars = reshape([var(:x, i)(value(t)) for i in eachindex(prob.u0)],size(prob.u0))
    params = p isa DiffEqBase.NullParameters ? [] :
             reshape([Num(Sym{Real}(nameof(Variable(:α, i)))) for i in eachindex(p)],size(p))

    D = Differential(t)

    rhs = [D(var) for var in vars]

    if DiffEqBase.isinplace(prob)
        lhs = similar(vars, Any)

        prob.f(lhs, vars, params, t)

        if DiffEqBase.is_diagonal_noise(prob)
            neqs = similar(vars, Any)
            prob.g(neqs, vars, params, t)
        else
            neqs = similar(vars, Any, size(prob.noise_rate_prototype))
            prob.g(neqs, vars, params, t)
        end
    else
        lhs = prob.f(vars, params, t)
        if DiffEqBase.is_diagonal_noise(prob)
            neqs = prob.g(vars, params, t)
        else
            neqs = prob.g(vars, params, t)
        end
    end
    deqs = vcat([rhs[i] ~ lhs[i] for i in eachindex(prob.u0)]...)

    de = SDESystem(deqs,neqs,t,vec(vars),vec(params))

    de
end


"""
$(TYPEDSIGNATURES)

Generate `OptimizationSystem`, dependent variables, and parameters from an `OptimizationProblem`.
"""
function modelingtoolkitize(prob::DiffEqBase.OptimizationProblem)

    if prob.p isa Tuple || prob.p isa NamedTuple
        p = [x for x in prob.p]
    else
        p = prob.p
    end

    vars = reshape([Num(Sym{Real}(nameof(Variable(:x, i)))) for i in eachindex(prob.u0)],size(prob.u0))
    params = p isa DiffEqBase.NullParameters ? [] :
             reshape([Num(Sym{Real}(nameof(Variable(:α, i)))) for i in eachindex(p)],size(Array(p)))

    eqs = prob.f(vars, params)
    de = OptimizationSystem(eqs,vec(vars),vec(params))
    de
end
