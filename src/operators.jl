struct Differential <: AbstractOperator
    x::Union{Variable,Operation}
    order::Int
end
Differential(x) = Differential(x,1)

Base.show(io::IO, D::Differential) = print(io,"($(D.x),$(D.order))")
Base.Expr(D::Differential) = :($(Symbol("D_$(D.x.name)_$(D.order)")))

function Derivative end
Base.:*(D::Differential,x::Operation) = Operation(Derivative,Expression[x,D])
function Base.:*(D::Differential,x::Variable)
    if D.x === x
        return Constant(1)
    else
        return Variable(x.name,x.subtype,x.value,x.value_type,D)
    end
end
Base.:(==)(D1::Differential, D2::Differential) = D1.order == D2.order && D1.x == D2.x

"""
expand_derivatives(O::Operation,constant_vars=[:DependentVariable])

Expands the derivative operation on the operation, applying the chain rule
and symbolically transforming the operation recrusively.
"""
function expand_derivatives(O::Operation)
    if O.op == Derivative
        #=
        diff_idxs = find(x->isequal(x,by.x),O.args)
        (diff_idxs != nothing || length(diff_idxs) > 1) && error("Derivatives of multi-argument functions require matching a unique argument.")
        idx = first(diff_idxs)
        =#
        i = 1
        if typeof(O.args[1].args[i]) == typeof(O.args[2].x) && isequal(O.args[1].args[i],O.args[2].x)
            Derivative(O.args[1],i)
        else
            D = Differential(O.args[2].x)
            cr_exp = D*O.args[1].args[i]
            Derivative(O.args[1],i) * expand_derivatives(cr_exp)
        end
    else
        for i in 1:length(O.args)
            O.args[i] = expand_derivatives(O.args[i])
        end
    end
end
expand_derivatives(x::Variable) = x

# Don't specialize on the function here
function Derivative(O::Operation,idx)
    # This calls the Derivative dispatch from the user or pre-defined code
    Derivative(O.op,O.args,Val{idx})
end

# Pre-defined derivatives
import DiffRules, SpecialFunctions, NaNMath
for (modu, fun, arity) in DiffRules.diffrules()
    if arity ==  1 && !(fun in (:-, :+)) # :+ and :- are both unary and binary operators
        @eval begin
            function Derivative(::typeof($modu.$fun), arg, ::Type{Val{1}})
                M, f = $(modu, fun)
                @assert length(arg) == 1 "$M.$f is a unary function!"
                dx = DiffRules.diffrule(M, f, arg[1])
                parse(Operation,dx)
            end
        end
    elseif arity ==  2
        for i in 1:2
            @eval begin
                function Derivative(::typeof($modu.$fun), args, ::Type{Val{$i}})
                    M, f =  $(modu, fun)
                    if f in (:-, :+)
                        @assert length(args) in (1, 2) "$M.$f is a unary or a binary function!"
                    else
                        @assert length(args) == 2 "$M.$f is a binary function!"
                    end
                    dx = DiffRules.diffrule(M, f, args[1], args[2])[$i]
                    parse(Operation,dx)
                end
            end
        end
    end
end

function count_order(x)
    @assert !(x isa Symbol) "The variable $x must have an order of differentiation that is greater or equal to 1!"
    n = 1
    while !(x.args[1] isa Symbol)
        n = n+1
        x = x.args[1]
    end
    n, x.args[1]
end

function _differetial_macro(x)
    ex = Expr(:block)
    lhss = Symbol[]
    x = flatten_expr!(x)
    for di in x
        @assert di isa Expr && di.args[1] == :~ "@Deriv expects a form that looks like `@Deriv D''~t E'~t`"
        lhs = di.args[2]
        rhs = di.args[3]
        order, lhs = count_order(lhs)
        push!(lhss, lhs)
        expr = :($lhs = Differential($rhs, $order))
        push!(ex.args,  expr)
    end
    push!(ex.args, Expr(:tuple, lhss...))
    ex
end

macro Deriv(x...)
    esc(_differetial_macro(x))
end

function calculate_jacobian(eqs,vars)
    Expression[Differential(vars[j])*eqs[i] for i in 1:length(eqs), j in 1:length(vars)]
end

export Differential, Derivative, expand_derivatives, @Deriv, calculate_jacobian
