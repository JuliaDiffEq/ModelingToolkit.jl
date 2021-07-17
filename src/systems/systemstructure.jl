module SystemStructures

using DataStructures
using SymbolicUtils: istree, operation, arguments, Symbolic
using ..ModelingToolkit
import ..ModelingToolkit: isdiffeq, var_from_nested_derivative, vars!, flatten,
    value, InvalidSystemException, isdifferential, _iszero, isparameter, independent_variables
using ..BipartiteGraphs
using LightGraphs
using UnPack
using Setfield
using SparseArrays

#=
When we don't do subsitution, variable information is split into two different
places, i.e. `states` and the right-hand-side of `observed`.

eqs = [0 ~ z + x; 0 ~ y + z^2]
states = [y, z]
observed = [x ~ sin(y) + z]
struct Reduced
    var
    expr
    idxs
end
fullvars = [Reduced(x, sin(y) + z, [2, 3]), y, z]
active_𝑣vertices = [false, true, true]
      x   y   z
eq1:  1       1
eq2:      1   1

      x   y   z
eq1:      1   1
eq2:      1   1

for v in 𝑣vertices(graph); active_𝑣vertices[v] || continue

end
=#

export SystemStructure, SystemPartition
export initialize_system_structure, find_linear_equations
export isdiffvar, isdervar, isalgvar, isdiffeq, isalgeq
export dervars_range, diffvars_range, algvars_range

@enum VariableType::Int8 DIFFERENTIAL_VARIABLE ALGEBRAIC_VARIABLE DERIVATIVE_VARIABLE

Base.@kwdef struct SystemPartition
    e_solved::Vector{Int}
    v_solved::Vector{Int}
    e_residual::Vector{Int}
    v_residual::Vector{Int}
end

function Base.:(==)(s1::SystemPartition, s2::SystemPartition)
    tup1 = (s1.e_solved, s1.v_solved, s1.e_residual, s1.v_residual)
    tup2 = (s2.e_solved, s2.v_solved, s2.e_residual, s2.v_residual)
    tup1 == tup2
end

Base.@kwdef struct SystemStructure
    fullvars::Vector
    vartype::Vector{VariableType}
    varassoc::Vector{Int}
    inv_varassoc::Vector{Int}
    varmask::BitVector # `true` if the variable has the highest order derivative
    algeqs::BitVector
    graph::BipartiteGraph{Int,Vector{Vector{Int}},Int,Nothing}
    solvable_graph::BipartiteGraph{Int,Vector{Vector{Int}},Int,Nothing}
    assign::Vector{Int}
    inv_assign::Vector{Int}
    scc::Vector{Vector{Int}}
    partitions::Vector{SystemPartition}
end

isdervar(s::SystemStructure, var::Integer) = s.vartype[var] === DERIVATIVE_VARIABLE
isdiffvar(s::SystemStructure, var::Integer) = s.vartype[var] === DIFFERENTIAL_VARIABLE
isalgvar(s::SystemStructure, var::Integer) = s.vartype[var] === ALGEBRAIC_VARIABLE

dervars_range(s::SystemStructure) = Iterators.filter(Base.Fix1(isdervar, s), eachindex(s.vartype))
diffvars_range(s::SystemStructure) = Iterators.filter(Base.Fix1(isdiffvar, s), eachindex(s.vartype))
algvars_range(s::SystemStructure) = Iterators.filter(Base.Fix1(isalgvar, s), eachindex(s.vartype))

isalgeq(s::SystemStructure, eq::Integer) = s.algeqs[eq]
isdiffeq(s::SystemStructure, eq::Integer) = !isalgeq(s, eq)

function initialize_system_structure(sys)
    sys = flatten(sys)
    ivs = independent_variables(sys)
    eqs = copy(equations(sys))
    neqs = length(eqs)
    algeqs = trues(neqs)
    dervaridxs = OrderedSet{Int}()
    var2idx = Dict{Any,Int}()
    symbolic_incidence = []
    fullvars = []
    var_counter = Ref(0)
    addvar! = let fullvars=fullvars, var_counter=var_counter
        var -> begin
            get!(var2idx, var) do
                push!(fullvars, var)
                var_counter[] += 1
            end
        end
    end

    vars = OrderedSet()
    for (i, eq′) in enumerate(eqs)
        if _iszero(eq′.lhs)
            eq = eq′
        else
            eq = 0 ~ eq′.rhs - eq′.lhs
        end
        vars!(vars, eq.rhs)
        isalgeq = true
        statevars = []
        for var in vars
            any(isequal(var), ivs) && continue
            if isparameter(var) || (istree(var) && isparameter(operation(var)))
                continue
            end
            varidx = addvar!(var)
            push!(statevars, var)

            dvar = var
            idx = varidx
            while isdifferential(dvar)
                if !(idx in dervaridxs)
                    push!(dervaridxs, idx)
                end
                isalgeq = false
                dvar = arguments(dvar)[1]
                idx = addvar!(dvar)
            end
        end
        push!(symbolic_incidence, copy(statevars))
        empty!(statevars)
        empty!(vars)
        algeqs[i] = isalgeq
        if isalgeq
            eqs[i] = eq
        end
    end

    # sort `fullvars` such that the mass matrix is as diagonal as possible.
    dervaridxs = collect(dervaridxs)
    sorted_fullvars = OrderedSet(fullvars[dervaridxs])
    for dervaridx in dervaridxs
        dervar = fullvars[dervaridx]
        diffvar = arguments(dervar)[1]
        if !(diffvar in sorted_fullvars)
            push!(sorted_fullvars, diffvar)
        end
    end
    for v in fullvars
        if !(v in sorted_fullvars)
            push!(sorted_fullvars, v)
        end
    end
    fullvars = collect(sorted_fullvars)
    var2idx = Dict(fullvars .=> eachindex(fullvars))
    dervaridxs = 1:length(dervaridxs)

    nvars = length(fullvars)
    diffvars = []
    vartype = fill(DIFFERENTIAL_VARIABLE, nvars)
    varassoc = zeros(Int, nvars)
    inv_varassoc = zeros(Int, nvars)
    for dervaridx in dervaridxs
        vartype[dervaridx] = DERIVATIVE_VARIABLE
        dervar = fullvars[dervaridx]
        diffvar = arguments(dervar)[1]
        diffvaridx = var2idx[diffvar]
        push!(diffvars, diffvar)
        varassoc[diffvaridx] = dervaridx
        inv_varassoc[dervaridx] = diffvaridx
    end

    algvars = setdiff(states(sys), diffvars)
    for algvar in algvars
        # it could be that a variable appeared in the states, but never appeared
        # in the equations.
        algvaridx = get(var2idx, algvar, 0)
        algvaridx == 0 && throw(InvalidSystemException("The system is missing "
            * "an equation for $algvar."
        ))
        vartype[algvaridx] = ALGEBRAIC_VARIABLE
    end

    graph = BipartiteGraph(neqs, nvars, Val(false))
    for (ie, vars) in enumerate(symbolic_incidence), v in vars
        jv = var2idx[v]
        add_edge!(graph, ie, jv)
    end

    @set! sys.eqs = eqs
    @set! sys.structure = SystemStructure(
        fullvars = fullvars,
        vartype = vartype,
        varassoc = varassoc,
        inv_varassoc = inv_varassoc,
        varmask = iszero.(varassoc),
        algeqs = algeqs,
        graph = graph,
        solvable_graph = BipartiteGraph(nsrcs(graph), ndsts(graph), Val(false)),
        assign = Int[],
        inv_assign = Int[],
        scc = Vector{Int}[],
        partitions = SystemPartition[],
    )
    return sys
end

function find_linear_equations(sys)
    s = structure(sys)
    @unpack fullvars, graph = s
    is_linear_equations = falses(nsrcs(graph))
    eqs = equations(sys)
    eadj = Vector{Int}[]
    cadj = Vector{Int}[]
    coeffs = Int[]
    for (i, eq) in enumerate(eqs); isdiffeq(eq) && continue
        empty!(coeffs)
        linear_term = 0
        all_int_vars = true

        term = value(eq.rhs - eq.lhs)
        for j in 𝑠neighbors(graph, i)
            var = fullvars[j]
            c = expand_derivatives(Differential(var)(term), false)
            # test if `var` is linear in `eq`.
            if !(c isa Symbolic) && c isa Number
                if c == 1 || c == -1
                    c = convert(Integer, c)
                    linear_term += c * var
                    push!(coeffs, c)
                else
                    all_int_vars = false
                end
            end
        end

        # Check if all states in the equation is both linear and homogeneous,
        # i.e. it is in the form of
        #
        #       ``∑ c_i * v_i = 0``,
        #
        # where ``c_i`` ∈ ℤ and ``v_i`` denotes states.
        if all_int_vars && isequal(linear_term, term)
            is_linear_equations[i] = true
            push!(eadj, copy(𝑠neighbors(graph, i)))
            push!(cadj, copy(coeffs))
        else
            is_linear_equations[i] = false
        end
    end

    return is_linear_equations, eadj, cadj
end

function Base.show(io::IO, s::SystemStructure)
    @unpack graph = s
    S = incidence_matrix(graph, Num(Sym{Real}(:×)))
    print(io, "Incidence matrix:")
    show(io, S)
end

end # module
