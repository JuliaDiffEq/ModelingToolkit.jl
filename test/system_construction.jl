using ModelingToolkit
using Base.Test

# Define some variables
@IVar t
@DVar x(t) y(t) z(t)
@Deriv D'~t # Default of first derivative, Derivative(t,1)
@Param σ ρ β
@Const c=0
@Var a

# Define a differential equation
eqs = [D*x ~ σ*(y-x),
       D*y ~ x*(ρ-z)-y,
       D*z ~ x*y - β*z]
de = DiffEqSystem(eqs,[t],[x,y,z],Variable[],[σ,ρ,β])
ModelingToolkit.generate_ode_function(de)
jac_expr = ModelingToolkit.generate_ode_jacobian(de)
jac = ModelingToolkit.calculate_jacobian(de)
jac_dvars = ModelingToolkit.calculate_jacobian(de,de.dvs)
f = DiffEqFunction(de)
W = I - jac
simplify_constants.(inv(W))

# Differential equation with automatic extraction of variables on rhs
de2 = DiffEqSystem(eqs, [t])
@test jac == jac_dvars

function test_vars_extraction(de, de2)
    for el in (:ivs, :dvs, :vs, :ps)
        names2 = sort(collect(var.name for var in getfield(de2,el)))
        names = sort(collect(var.name for var in getfield(de,el)))
        @test names2 == names
    end
end
test_vars_extraction(de, de2)

#Lotka Volterra test for jacobians
eqs = [D*x ~ a*x-b*x*y,
       D*y ~c*x*y-d*y]
sys = DiffEqSystem(eqs,[t],[x,y],Variable[],[a,b,c,d])
param_jac = ModelingToolkit.calculate_jacobian(de,de.ps)
dvars_jac = ModelingToolkit.calculate_jacobian(de,de.dvs)
@test param_jac == [y - x  Constant(0)  Constant(0); Constant(0)  DependentVariable(x)  Constant(0);Constant(0) Constant(0) -1 * z]
@test dvars_jac == [σ * -1 Parameter(σ)  Constant(0);ρ - z Constant(-1) x * -1;DependentVariable(y)  DependentVariable(x) -1 * β]


# Conversion to first-order ODEs #17
@Deriv D3'''~t
@Deriv D2''~t
@DVar u(t) u_tt(t) u_t(t) x_t(t)
eqs = [D3*u ~ 2(D2*u) + D*u + D*x + 1
       D2*x ~ D*x + 2]
de = DiffEqSystem(eqs, [t])
de1 = ode_order_lowering(de)
lowered_eqs = [D*u_tt ~ 2u_tt + u_t + x_t + 1
               D*x_t  ~ x_t + 2
               D*u_t  ~ u_tt
               D*u    ~ u_t
               D*x    ~ x_t]
function test_eqs(eqs1, eqs2)
    eq = true
    for i in eachindex(eqs1)
        lhs1, lhs2 = eqs1[i].args[1], eqs2[i].args[1]
        for f in fieldnames(typeof(lhs1))
            eq = eq && isequal(getfield(lhs1, f), getfield(lhs2, f))
        end
        eq = eq && isequal(eqs1[i].args[2], eqs2[i].args[2])
    end
    @test_broken eq
end
test_eqs(de1.eqs, lowered_eqs)

# Internal calculations
eqs = [a ~ y-x,
       D*x ~ σ*a,
       D*y ~ x*(ρ-z)-y,
       D*z ~ x*y - β*z]
de = DiffEqSystem(eqs,[t],[x,y,z],[a],[σ,ρ,β])
ModelingToolkit.generate_ode_function(de)
jac = ModelingToolkit.calculate_jacobian(de)
f = DiffEqFunction(de)

# Define a nonlinear system
eqs = [0 ~ σ*(y-x),
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]
ns = NonlinearSystem(eqs,[x,y,z],[σ,ρ,β])
ns2 = NonlinearSystem(eqs)
for el in (:vs, :ps)
    names2 = sort(collect(var.name for var in getfield(ns2,el)))
    names = sort(collect(var.name for var in getfield(ns,el)))
    @test names2 == names
end

ModelingToolkit.generate_nlsys_function(ns)

# Now nonlinear system with only variables
@Var x y z
@Param σ ρ β

# Define a nonlinear system
eqs = [0 ~ σ*(y-x),
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]
ns = NonlinearSystem(eqs)
nlsys_func = ModelingToolkit.generate_nlsys_function(ns)
jac = ModelingToolkit.generate_nlsys_jacobian(ns)
f = @eval eval(nlsys_func)

# Intermediate calculations
# Define a nonlinear system
eqs = [a ~ y-x,
       0 ~ σ*a,
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]
ns = NonlinearSystem(eqs,[x,y,z],[σ,ρ,β])
nlsys_func = ModelingToolkit.generate_nlsys_function(ns)
jac = ModelingToolkit.calculate_jacobian(ns)
jac = ModelingToolkit.generate_nlsys_jacobian(ns)
