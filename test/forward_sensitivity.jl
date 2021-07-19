using ModelingToolkit
@parameters t #time
@derivatives D'~t
@variables Cₛ(t), Cₓ(t), ν(t) # states
x = [Cₛ, Cₓ, ν]
n_x = length(x)
@parameters uᵢₙ # control input
u = [uᵢₙ]
n_u = length(u)
@parameters μₘₐₓ Cₛ_ᵢₙ Cₛ₀ Cₓ₀ # uncertain parameters
p = [μₘₐₓ, Cₛ_ᵢₙ, Cₛ₀, Cₓ₀]
n_p = length(p)
u₀ = [Cₛ₀, Cₓ₀, μₘₐₓ] # note third initial condition is also present in the dynamics
@parameters  Kₛ, yₓ_ₛ, m #other constants
μ = μₘₐₓ*Cₛ/Kₛ + Cₛ
σ = μ/yₓ_ₛ + m

eqs = [D(Cₛ) ~ -σ*Cₓ + uᵢₙ/ν*Cₛ_ᵢₙ - uᵢₙ/ν*Cₛ,
       D(Cₓ) ~ μ*Cₓ - uᵢₙ/ν*Cₓ,
       D(ν) ~ uᵢₙ]
sys = ODESystem(eqs) 
length(sys.states)
length(sys.eqs)

# TODO: actual tests
sysₑₓₜ = ModelingToolkit.forward_sensitivity_transform(sys,p)
u₀ₑₓₜ = ModelingToolkit.forward_sensitivity_intial_condition(u₀,p)

partial_p = [p[1],p[3]]
sysₑₓₜ = ModelingToolkit.forward_sensitivity_transform(sys,partial_p)
u₀ₑₓₜ = ModelingToolkit.forward_sensitivity_intial_condition(u₀,partial_p)
