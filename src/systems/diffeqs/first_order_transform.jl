"""
$(TYPEDSIGNATURES)

Takes a Nth order ODESystem and returns a new ODESystem written in first order
form by defining new variables which represent the N-1 derivatives.
"""
function ode_order_lowering(sys::ODESystem)
    iv = get_iv(sys)
    eqs_lowered, new_vars = ode_order_lowering(equations(sys), iv, states(sys))
    @set! sys.eqs = eqs_lowered
    @set! sys.states = new_vars
    @set! sys.structure = nothing
    return sys
end

function ode_order_lowering(eqs, iv, states)
    var_order = OrderedDict{Any,Int}()
    D = Differential(iv)
    diff_eqs = Equation[]
    diff_vars = []
    alge_eqs = Equation[]

    for (i, eq) ∈ enumerate(eqs)
        if !isdiffeq(eq)
            push!(alge_eqs, eq)
        else
            var, maxorder = var_from_nested_derivative(eq.lhs)
            maxorder > get(var_order, var, 1) && (var_order[var] = maxorder)
            var′ = lower_varname(var, iv, maxorder - 1)
            rhs′ = diff2term(eq.rhs)
            push!(diff_vars, var′)
            push!(diff_eqs, D(var′) ~ rhs′)
        end
    end

    for (var, order) ∈ var_order
        for o in (order-1):-1:1
            lvar = lower_varname(var, iv, o-1)
            rvar = lower_varname(var, iv, o)
            push!(diff_vars, lvar)

            rhs = rvar
            eq = Differential(iv)(lvar) ~ rhs
            push!(diff_eqs, eq)
        end
    end

    # we want to order the equations and variables to be `(diff, alge)`
    return (vcat(diff_eqs, alge_eqs), vcat(diff_vars, setdiff(states, diff_vars)))
end
