using ModelingToolkit, LinearAlgebra, DiffEqJump, Test, Distributions
MT = ModelingToolkit

@parameters t k[1:20]
@variables A(t) B(t) C(t) D(t)
rxs = [Reaction(k[1], nothing, [A]),            # 0 -> A
       Reaction(k[2], [B], nothing),            # B -> 0
       Reaction(k[3],[A],[C]),                  # A -> C
       Reaction(k[4], [C], [A,B]),              # C -> A + B
       Reaction(k[5], [C], [A], [1], [2]),      # C -> A + A
       Reaction(k[6], [A,B], [C]),              # A + B -> C
       Reaction(k[7], [B], [A], [2], [1]),      # 2B -> A
       Reaction(k[8], [A,B], [A,C]),            # A + B -> A + C
       Reaction(k[9], [A,B], [C,D]),            # A + B -> C + D
       Reaction(k[10], [A], [C,D], [2], [1,1]), # 2A -> C + D
       Reaction(k[11], [A], [A,B], [2], [1,1]), # 2A -> A + B
       Reaction(k[12], [A,B], nothing, [3,1], nothing),           # 3A+B -> 0
       Reaction(k[13], [A,B,C], [C,D], [1,3,4], [Poisson(2), 3]),          # A+3B+4C -> k_pois * C + 3D
       Reaction(k[14], nothing, [A], nothing, [Geometric(1 // 3)]),               # 0 -> k_geom * A
       Reaction(k[15]*A/(2+A), [A], nothing; only_use_rate=true), # A -> 0 with custom rate
       Reaction(k[16], [A], [B]; only_use_rate=true),             # A -> B with custom rate.
       Reaction(k[17]*A*exp(B), [C], [D], [2], [Bernoulli(1)]),   # 2C -> D with non constant rate.
       Reaction(k[18]*B, nothing, [B], nothing, [2]),             # 0 -> 2B with non constant rate.
       Reaction(k[19]*t, [A], [B]),                                # A -> B with non constant rate.
       Reaction(k[20]*t*A, [B,C], [D],[2,1],[2])                  # 2A +B -> 2C with non constant rate.
  ]
rs = ReactionSystem(rxs,t,[A,B,C,D],k)
odesys = convert(ODESystem,rs)
sdesys = convert(SDESystem,rs)

# test show
io = IOBuffer()
show(io, rs)
str = String(take!(io))
@test count(isequal('\n'), str) < 30

# hard coded ODE rhs
function oderhs(u,k,t)
       A = u[1]; B = u[2]; C = u[3]; D = u[4];
       du = zeros(eltype(u),4)
       du[1] = k[1] - k[3]*A + k[4]*C + 2*k[5]*C - k[6]*A*B + k[7]*B^2/2 - k[9]*A*B - k[10]*A^2 - k[11]*A^2/2 - k[13]*A*B^3*C^4/144 - 3*k[12]*A^3*B/6 + 2*k[14] - k[15]*A/(2+A) - k[16] - k[19]*t*A
       du[2] = -k[2]*B + k[4]*C - k[6]*A*B - k[7]*B^2 - k[8]*A*B - k[9]*A*B + k[11]*A^2/2 - 3*k[13]*A*B^3*C^4/144 - k[12]*A^3*B/6 + k[16] + 2*k[18]*B + k[19]*t*A - 2*k[20]*t*A*B^2*C
       du[3] = k[3]*A - k[4]*C - k[5]*C  + k[6]*A*B + k[8]*A*B + k[9]*A*B + k[10]*A^2/2 - 2*k[13]*A*B^3*C^4/144 - 2*k[17]*A*exp(B)*C^2/2 - k[20]*t*A*B^2*C
       du[4] = k[9]*A*B + k[10]*A^2/2 + 3*k[13]*A*B^3*C^4/144 + k[17]*A*exp(B)*C^2/2 + 2*k[20]*t*A*B^2*C
       du
end

# sde noise coefs
function sdenoise(u,k,t)
       A = u[1]; B = u[2]; C = u[3]; D = u[4];
       G = zeros(eltype(u),length(k),length(u))
       z = zero(eltype(u))

       G = [sqrt(k[1]) z  z z;
            z -sqrt(k[2]*B) z z;
            -sqrt(k[3]*A) z sqrt(k[3]*A) z;
            sqrt(k[4]*C) sqrt(k[4]*C) -sqrt(k[4]*C) z;
            2*sqrt(k[5]*C) z -sqrt(k[5]*C) z;
            -sqrt(k[6]*A*B) -sqrt(k[6]*A*B) sqrt(k[6]*A*B) z;
            sqrt(k[7]*B^2/2) -2*sqrt(k[7]*B^2/2) z z;
            z -sqrt(k[8]*A*B) sqrt(k[8]*A*B) z;
            -sqrt(k[9]*A*B) -sqrt(k[9]*A*B) sqrt(k[9]*A*B) sqrt(k[9]*A*B);
            -2*sqrt(k[10]*A^2/2) z sqrt(k[10]*A^2/2) sqrt(k[10]*A^2/2);
            -sqrt(k[11]*A^2/2) sqrt(k[11]*A^2/2) z z;
            -3*sqrt(k[12]*A^3*B/6) -sqrt(k[12]*A^3*B/6) z z;
            -sqrt(k[13]*A*B^3*C^4/144) -3*sqrt(k[13]*A*B^3*C^4/144) -2*sqrt(k[13]*A*B^3*C^4/144) 3*sqrt(k[13]*A*B^3*C^4/144);
            2*sqrt(k[14]) z z z;
            -sqrt(k[15]*A/(2+A)) z z z;
            -sqrt(k[16]) sqrt(k[16]) z z;
            z z -2*sqrt(k[17]*A*exp(B)*C^2/2) sqrt(k[17]*A*exp(B)*C^2/2);
            z 2*sqrt(k[18]*B) z z;
            -sqrt(k[19]*t*A) sqrt(k[19]*t*A) z z;
            z -2*sqrt(k[20]*t*A*B^2*C) -sqrt(k[20]*t*A*B^2*C) +2*sqrt(k[20]*t*A*B^2*C);
            # burst variances
            z z sqrt(k[13]*A*B^3*C^4/144) * sqrt(2.) z;
            sqrt(k[14])*sqrt(6.) z z z;
            z z z z]';
    
       return G
end

# test by evaluating drift terms
p  = rand(length(k))
u  = rand(length(k))
t  = 0.
du = oderhs(u,p,t)
odesys = convert(ODESystem,rs)
of = ODEFunction{false}(odesys, states(rs), parameters(rs))
du2 = of.f(u,p,t)
@test norm(du-du2) < 100*eps()

# test by evaluating drift and diffusion terms
p  = rand(length(k))
u  = rand(length(k))
t  = 0.
du = oderhs(u,p,t)
G  = sdenoise(u,p,t)
sdesys = convert(SDESystem,rs)
sf = SDEFunction{false}(sdesys, states(rs), parameters(rs))
du2 = sf.f(u,p,t)
@test norm(du-du2) < 100*eps()
G2 = sf.g(u,p,t)
@test norm(G-G2) < 100*eps()

# test conversion to NonlinearSystem
ns = convert(NonlinearSystem,rs)
fnl = eval(generate_function(ns)[2])
dunl = similar(du)
fnl(dunl,u,p)
@test norm(du-dunl) < 100*eps()

# tests the noise_scaling argument.
p  = rand(length(k)+1)
u  = rand(length(k))
t  = 0.
G  = p[21]*sdenoise(u,p,t)
@variables η
sdesys_noise_scaling = convert(SDESystem,rs;noise_scaling=η)
sf = SDEFunction{false}(sdesys_noise_scaling, states(rs), parameters(sdesys_noise_scaling))
G2 = sf.g(u,p,t)
@test norm(G-G2) < 100*eps()

# tests the noise_scaling vector argument.
p  = rand(length(k)+3)
u  = rand(length(k))
t  = 0.
G  = vcat(fill(p[21],8),fill(p[22],5),fill(p[23],7),[p[22],p[23],p[23]])' .* sdenoise(u,p,t)
@variables η[1:3]
sdesys_noise_scaling = convert(SDESystem,rs;noise_scaling=vcat(fill(η[1],8),fill(η[2],5),fill(η[3],7)))
sf = SDEFunction{false}(sdesys_noise_scaling, states(rs), parameters(sdesys_noise_scaling))
G2 = sf.g(u,p,t)
@test norm(G-G2) < 100*eps()

# tests using previous parameter for noise scaling
p  = rand(length(k)+3)
p[21], p[22], p[23] = p[13], p[14], p[17]
u  = rand(length(k))
t  = 0.
G  = [p p p p]' .* sdenoise(u,p,t)
sdesys_noise_scaling = convert(SDESystem,rs;noise_scaling=vcat(k))
sf = SDEFunction{false}(sdesys_noise_scaling, states(rs), parameters(sdesys_noise_scaling))
G2 = sf.g(u,p,t)
@test norm(G-G2) < 100*eps()

# test with JumpSystem
js = convert(JumpSystem, rs)

midxs = 1:12
cidxs = 13:18
vidxs = 19:20
@test all(map(i -> typeof(equations(js)[i]) <: DiffEqJump.MassActionJump, midxs))
@test all(map(i -> typeof(equations(js)[i]) <: DiffEqJump.ConstantRateJump, cidxs))
@test all(map(i -> typeof(equations(js)[i]) <: DiffEqJump.VariableRateJump, vidxs))

pars = rand(length(k)); u0 = rand(1:10,4); ttt = rand();
jumps = Vector{Union{ConstantRateJump, MassActionJump, VariableRateJump}}(undef,length(rxs))

jumps[1] = MassActionJump(pars[1], Vector{Pair{Int,Int}}(), [1 => 1]);
jumps[2] = MassActionJump(pars[2], [2 => 1], [2 => -1]);
jumps[3] = MassActionJump(pars[3], [1 => 1], [1 => -1, 3 => 1]);
jumps[4] = MassActionJump(pars[4], [3 => 1], [1 => 1, 2 => 1, 3 => -1]);
jumps[5] = MassActionJump(pars[5], [3 => 1], [1 => 2, 3 => -1]);
jumps[6] = MassActionJump(pars[6], [1 => 1, 2 => 1], [1 => -1, 2 => -1, 3 => 1]);
jumps[7] = MassActionJump(pars[7], [2 => 2], [1 => 1, 2 => -2]);
jumps[8] = MassActionJump(pars[8], [1 => 1, 2 => 1], [2 => -1, 3 => 1]);
jumps[9] = MassActionJump(pars[9], [1 => 1, 2 => 1], [1 => -1, 2 => -1, 3 => 1, 4 => 1]);
jumps[10] = MassActionJump(pars[10], [1 => 2], [1 => -2, 3 => 1, 4 => 1]);
jumps[11] = MassActionJump(pars[11], [1 => 2], [1 => -1, 2 => 1]);
jumps[12] = MassActionJump(pars[12], [1 => 3, 2 => 1], [1 => -3, 2 => -1]);

jumps[13] = ConstantRateJump((u,p,t) -> p[13]*u[1]*binomial(u[2],3)*binomial(u[3],4), 
                              integrator -> (integrator.u[1] -= 1; integrator.u[2] -= 3; integrator.u[3] += 2 - 4; integrator.u[4]+=3));
jumps[14] = ConstantRateJump((u,p,t) -> p[14], integrator -> (integrator.u[1] += 1));

jumps[15] = ConstantRateJump((u,p,t) -> p[15]*u[1]/(2+u[1]), integrator -> (integrator.u[1] -= 1))
jumps[16] = ConstantRateJump((u,p,t) -> p[16], integrator -> (integrator.u[1] -= 1; integrator.u[2] += 1;))
jumps[17] = ConstantRateJump((u,p,t) -> p[17]*u[1]*exp(u[2])*binomial(u[3],2), integrator -> (integrator.u[3] -= 2; integrator.u[4] += 1))
jumps[18] = ConstantRateJump((u,p,t) -> p[18]*u[2], integrator -> (integrator.u[2] += 2))

jumps[19] = VariableRateJump((u,p,t) -> p[19]*u[1]*t, integrator -> (integrator.u[1] -= 1; integrator.u[2] += 1))
jumps[20] = VariableRateJump((u,p,t) -> p[20]*t*u[1]*binomial(u[2],2)*u[3], integrator -> (integrator.u[2] -= 2; integrator.u[3] -= 1; integrator.u[4] += 2))

statetoid = Dict(state => i for (i,state) in enumerate(states(js)))
parammap = map((x,y)->Pair(x,y),parameters(js),pars)
for i in midxs
  maj = MT.assemble_maj(equations(js)[i], statetoid, ModelingToolkit.substituter(parammap),eltype(pars))
  @test abs(jumps[i].scaled_rates - maj.scaled_rates) < 100*eps()
  @test jumps[i].reactant_stoch == maj.reactant_stoch
  @test jumps[i].net_stoch == maj.net_stoch
end
for i in cidxs
  crj = MT.assemble_crj(js, equations(js)[i], statetoid)
  @test isapprox(crj.rate(u0,p,ttt), jumps[i].rate(u0,p,ttt))
  fake_integrator1 = (u=zeros(4),p=p,t=0); fake_integrator2 = deepcopy(fake_integrator1);
  crj.affect!(fake_integrator1);
  jumps[i].affect!(fake_integrator2);
  # Won't have equality in general due to randomness in bursts
  #@test fake_integrator1 == fake_integrator2
end
for i in vidxs
  crj = MT.assemble_vrj(js, equations(js)[i], statetoid)
  @test isapprox(crj.rate(u0,p,ttt), jumps[i].rate(u0,p,ttt))
  fake_integrator1 = (u=zeros(4),p=p,t=0.); fake_integrator2 = deepcopy(fake_integrator1);
  crj.affect!(fake_integrator1); jumps[i].affect!(fake_integrator2);
  @test fake_integrator1 == fake_integrator2
end