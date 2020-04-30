struct Reaction{S <: Variable, T <: Number}
    rate
    substrates::Vector{Operation}
    products::Vector{Operation}
    substoich::Vector{T}
    prodstoich::Vector{T}
    netstoich::Vector{Pair{S,T}}
    only_use_rate::Bool
end

function Reaction(rate, subs, prods, substoich, prodstoich;
                  netstoich=nothing, only_use_rate=false, kwargs...)

    subsv  = (isempty(subs)||isnothing(subs)) ? Vector{Operation}() : subs
    prodsv = (isempty(prods)||isnothing(prods)) ? Vector{Operation}() : prods
    ns = isnothing(netstoich) ? get_netstoich(subsv, prodsv, substoich, prodstoich) : netstoich
    Reaction(rate, subsv, prodsv, substoich, prodstoich, ns, only_use_rate)
end

# three argument constructor assumes stoichiometric coefs are one and integers
function Reaction(rate, subs, prods; kwargs...)

    sstoich = isnothing(subs) ? Int[] : ones(Int,length(subs))
    pstoich = isnothing(prods) ? Int[] : ones(Int,length(prods))
    Reaction(rate, subs, prods, sstoich, pstoich; kwargs...)
end

# calculates the net stoichiometry of a reaction as a vector of pairs (sub,substoich)
function get_netstoich(subs, prods, sstoich, pstoich)
    # stoichiometry as a Dictionary
    nsdict = Dict{Variable,eltype(sstoich)}(sub.op => -sstoich[i] for (i,sub) in enumerate(subs))
    for (i,p) in enumerate(prods)
        coef = pstoich[i]
        prod = p.op
        @inbounds nsdict[prod] = haskey(nsdict, prod) ? nsdict[prod] + coef : coef
    end

    # stoichiometry as a vector
    ns = [el for el in nsdict if el[2] != zero(el[2])]

    ns
end


struct ReactionSystem <: AbstractSystem
    eqs::Vector{Reaction}
    iv::Variable
    states::Vector{Variable}
    ps::Vector{Variable}
    name::Symbol
    systems::Vector{ReactionSystem}
end

function ReactionSystem(eqs, iv, species, params; systems = ReactionSystem[],
                                                  name = gensym(:ReactionSystem))

    ReactionSystem(eqs, iv, convert.(Variable,species), convert.(Variable,params),
                   name, systems)
end

# Calculate the ODE rate law
function oderatelaw(rx)
    @unpack rate, substrates, substoich, only_use_rate = rx
    rl = rate
    if !only_use_rate
        coef = one(eltype(substoich))
        for (i,stoich) in enumerate(substoich)
            coef *= factorial(stoich)
            rl   *= isone(stoich) ? substrates[i] : substrates[i]^stoich
        end
        (!isone(coef)) && (rl /= coef)
    end
    rl
end

function assemble_drift(rs)
    D   = Differential(rs.iv())
    eqs = [D(x(rs.iv())) ~ 0 for x in rs.states]
    species_to_idx = Dict((x => i for (i,x) in enumerate(rs.states)))

    for rx in rs.eqs
        rl = oderatelaw(rx)
        for (spec,stoich) in rx.netstoich
            i = species_to_idx[spec]
            if iszero(eqs[i].rhs)
                signedrl = (stoich > zero(stoich)) ? rl : -rl
                rhs      = isone(abs(stoich)) ? signedrl : stoich * rl
            else
                Δspec = isone(abs(stoich)) ? rl : abs(stoich) * rl
                rhs   = (stoich > zero(stoich)) ? (eqs[i].rhs + Δspec) : (eqs[i].rhs - Δspec)
            end
            eqs[i] = Equation(eqs[i].lhs, rhs)
        end
    end
    eqs
end

function assemble_diffusion(rs)
    eqs = Expression[Constant(0) for x in rs.states, y in rs.eqs]
    species_to_idx = Dict((x => i for (i,x) in enumerate(rs.states)))

    for (j,rx) in enumerate(rs.eqs)
        rlsqrt = sqrt(oderatelaw(rx))
        for (spec,stoich) in rx.netstoich
            i            = species_to_idx[spec]
            signedrlsqrt = (stoich > zero(stoich)) ? rlsqrt : -rlsqrt
            eqs[i,j]     = isone(abs(stoich)) ? signedrlsqrt : stoich * rlsqrt
        end
    end
    eqs
end

function Base.convert(::Type{<:ODESystem},rs::ReactionSystem)
    eqs = assemble_drift(rs)
    ODESystem(eqs,rs.iv,rs.states,rs.ps,name=rs.name,
              systems=convert.(ODESystem,rs.systems))
end

function Base.convert(::Type{<:SDESystem},rs::ReactionSystem)
    eqs = assemble_drift(rs)
    noiseeqs = assemble_diffusion(rs)
    SDESystem(eqs,noiseeqs,rs.iv,rs.states,rs.ps,
              name=rs.name,systems=convert.(SDESystem,rs.systems))
end
