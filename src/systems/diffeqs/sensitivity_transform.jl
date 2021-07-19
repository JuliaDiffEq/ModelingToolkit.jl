function forward_sensitivity_transform(sys::ODESystem,p::AbstractVector{<:Expression})::ODESystem
    iv  = sys.iv()
    f = du_dt = [eq.rhs for eq ∈ sys.eqs]
    u = [state(iv) for state ∈ states(sys)] 
    D_p = [Differential(par.op.name) for par ∈ p]
    du_dp = [D_par(state) for state ∈ u, D_par ∈ D_p]
    D = Differential(iv.op.name)
    uₑₓₜ= vcat(u,vec(du_dp))
    df_du = jacobian(f,u)
    df_dp = jacobian(f,p)
    ddu_dpdt = df_du*du_dp + df_dp
    duₑₓₜ_dt = fₑₓₜ = vcat(du_dt,vec(ddu_dpdt))
    eqsₑₓₜ = simplify.(D.(uₑₓₜ) .~ fₑₓₜ)
    sysₑₓₜ = ODESystem(eqsₑₓₜ, iv, uₑₓₜ, sys.ps)
end

function forward_sensitivity_intial_condition(u₀::AbstractVector{<:Expression},p::AbstractVector{<:Expression})::AbstractVector{<:Expression}
    du₀_dp = jacobian(u₀,p)
    u₀ₑₓₜ = vcat(u₀,vec(du₀_dp))
end
