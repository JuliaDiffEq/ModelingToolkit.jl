mutable struct Variable <: Expression
    name::Symbol
    value
    value_type::DataType
    subtype::Symbol
    diff::Union{Function,Nothing}  # FIXME
    dependents::Vector{Variable}
    description::String
    flow::Bool
    domain::AbstractDomain
    size
    context
end

Variable(name,
         value = nothing,
         value_type = typeof(value);
         subtype::Symbol=:Variable,
         dependents::Vector{Variable} = Variable[],
         flow::Bool = false,
         description::String = "",
         domain = Reals(),
         size = nothing,
         context = nothing) =
         Variable(name,value,value_type,subtype,nothing,
                  dependents,description,flow,domain,size,context)
Variable(name,args...;kwargs...) = Variable(name,args...;subtype=:Variable,kwargs...)

Variable(name,x::Variable) = Variable(name,x.value,x.value_type,
                x.subtype,D,x.dependents,x.description,x.flow,x.domain,
                x.size,x.context)

Parameter(name,args...;kwargs...) = Variable(name,args...;subtype=:Parameter,kwargs...)
Constant(value::Number) = Variable(Symbol(value),value,typeof(value);subtype=:Constant)
Constant(name,args...;kwargs...) = Variable(name,args...;subtype=:Constant,kwargs...)
IndependentVariable(name,args...;kwargs...) = Variable(name,args...;subtype=:IndependentVariable,kwargs...)

function DependentVariable(name,args...;dependents = [],kwargs...)
    @assert !isempty(dependents)
    Variable(name,args...;subtype=:DependentVariable,dependents=dependents,kwargs...)
end

function StateVariable(name,args...;dependents = [],kwargs...)
    @assert !isempty(dependents)
    Variable(name,args...;subtype=:StateVariable,dependents=dependents,kwargs...)
end

function ControlVariable(name,args...;dependents = [],kwargs...)
    @assert !isempty(dependents)
    Variable(name,args...;subtype=:ControlVariable,dependents=dependents,kwargs...)
end

function JumpVariable(name,args...;dependents = [],kwargs...)
    @assert !isempty(dependents)
    Variable(name,args...;subtype=:JumpVariable,dependents=dependents,kwargs...)
end

function NoiseVariable(name,args...;dependents = [],kwargs...)
    @assert !isempty(dependents)
    Variable(name,args...;subtype=:NoiseVariable,dependents=dependents,kwargs...)
end

export Variable,Parameter,Constant,DependentVariable,IndependentVariable,JumpVariable,NoiseVariable,
       @Var, @DVar, @IVar, @Param, @Const


Base.get(x::Variable) = x.value

Base.iszero(::Expression) = false
Base.iszero(c::Variable) = get(c) isa Number && iszero(get(c))
Base.isone(::Expression) = false
Base.isone(c::Variable) = get(c) isa Number && isone(get(c))


# Variables use isequal for equality since == is an Operation
function Base.:(==)(x::Variable,y::Variable)
    x.name == y.name && x.subtype == y.subtype && x.value == y.value &&
    x.value_type == y.value_type && x.diff == y.diff
end

function Base.:(==)(x::Variable,y::Number)
    x == Constant(y)
end

function Base.:(==)(x::Number,y::Variable)
    Constant(x) == y
end

function Base.Expr(x::Variable)
    if x.subtype == :Constant
        return x.value
    elseif x.diff == nothing
        return :($(x.name))
    else
        return :($(Symbol("$(x.name)_$(x.diff.x.name)")))
    end
end

function Base.show(io::IO, A::Variable)
    if A.subtype == :Constant
        print(io,"Constant($(A.value))")
    else
        str = "$(A.subtype)($(A.name))"
        if A.value != nothing
            str *= ", value = " * string(A.value)
        end

        if A.diff != nothing
            str *= ", diff = " * string(A.diff)
        end

        print(io,str)
    end
end

# Build variables more easily
function _parse_vars(macroname, fun, x)
    ex = Expr(:block)
    lhss = Symbol[]
    # if parsing things in the form of
    # begin
    #     x
    #     y
    #     z = exp(2)
    # end
    x = flatten_expr!(x)
    for _var in x
        iscall = typeof(_var) <: Expr && _var.head == :call
        issym    = _var isa Symbol
        isassign = issym ? false : _var.head == :(=)
        @assert iscall || issym || isassign "@$macroname expects a tuple of expressions!\nE.g. `@$macroname x y z=1`"
        if iscall || issym
            if iscall
                dependents = :([$(_var.args[2:end]...)])
                var = _var.args[1]
            else
                dependents = Variable[]
                var = _var
            end
            lhs = var
            push!(lhss, lhs)
            expr = :( $lhs = $fun( Symbol($(String(lhs))) ,
                      dependents = $dependents))
        end
        if isassign
            iscall = typeof(_var.args[1]) <: Expr && _var.args[1].head == :call
            if iscall
                dependents = :([$(_var.args[1].args[2:end]...)])
                lhs = _var.args[1].args[1]
            else
                dependents = Variable[]
                lhs = _var.args[1]
            end
            rhs = _var.args[2]
            push!(lhss, lhs)
            expr = :( $lhs = $fun( Symbol($(String(lhs))) , $rhs,
                      dependents = $dependents))
        end
        push!(ex.args, expr)
    end
    push!(ex.args, Expr(:tuple, lhss...))
    ex
end

for funs in ((:DVar, :DependentVariable), (:IVar, :IndependentVariable),
             (:Var, :Variable),
             (:Param, :Parameter))
    @eval begin
        macro ($(funs[1]))(x...)
            esc(_parse_vars(String($funs[1]), $funs[2], x))
        end
    end
end

function _const_assign(x)
    ex = Expr(:block)
    lhss = Symbol[]
    x = flatten_expr!(x)
    for eq in x
        @assert eq isa Expr && eq.head == :(=) "@Const expects a tuple of assignments!\nE.g. `@Const D=t W=g`"
        lhs = eq.args[1]
        push!(lhss, lhs)
        rhs = eq.args[2]
        expr = :($lhs = Constant($rhs))
        push!(ex.args,  expr)
    end
    push!(ex.args, Expr(:tuple, lhss...))
    ex
end

macro Const(x...)
    esc(_const_assign(x))
end
