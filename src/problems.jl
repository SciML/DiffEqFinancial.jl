@doc doc"""

```math
dS = μSdt + \sqrt{v}SdW_1
dv = κ(Θ-v)dt + σ\sqrt{v}dW_2
dW_1 dW_2 = ρ dt
```

"""
function HestonProblem(μ,κ,Θ,σ,ρ,u0,tspan;seed=UInt64(0),kwargs...)
  f = function (du,u,p,t)
    du[1] = μ*u[1]
    du[2] = κ*(Θ-u[2])
  end
  g = function (du,u,p,t)
    du[1] = √u[2]*u[1]
    du[2] = Θ*√u[2]
  end
  Γ = [1 ρ;ρ 1] # Covariance Matrix
  noise_rate_prototype = nothing

  if seed == 0
    seed = rand(UInt64)
  end
  noise = CorrelatedWienerProcess!(Γ,tspan[1],zeros(2),zeros(2),
                                   rng = Xorshifts.Xoroshiro128Plus(seed))

  sde_f = SDEFunction{true}(f,g)
  SDEProblem(sde_f,g,u0,tspan,noise=noise,seed=seed,kwargs...)
end

@doc doc"""

``d \ln S(t) = (r(t) - q(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
function GeneralizedBlackScholesProblem(r,q,Θ,σ,u0,tspan;kwargs...)
  f = function (u,p,t)
    r(t) - q(t) - Θ(t,exp(u))^2 / 2
  end
  g = function (u,p,t)
    σ
  end
  SDEProblem{false}(f,g,u0,tspan;kwargs...)
end

@doc doc"""

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
mutable struct BlackScholesProblem{uType,tType,tupType,isinplace,NP,F,F2,C,ND,MM} <: DiffEqBase.AbstractSDEProblem{uType,tType,isinplace,ND}
  r::tType
  Θ::tType
  σ::tType
  u0::uType
  tspan::tupType
  p::Nothing
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
  seed::UInt64
end

BlackScholesProblem(r,Θ,σ,u0,tspan;callback = CallbackSet(),
                    noise_rate_prototype = nothing,seed=UInt64(0)) =
                    GeneralizedBlackScholesProblem(r,(t)->0,Θ,σ,u0,tspan,callback=callback,seed=seed)

@doc doc"""

``dx = a(b(t)-x)dt + σ dW_t``

"""
function ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u0,tspan;kwargs...)
  f = function (u,p,t)
    a*(b(t)-u)
  end
  g = function (u,p,t)
    σ
  end
  SDEProblem{false}(f,g,u0,tspan;kwargs...)
end

@doc doc"""

``dx = a(r-x)dt + σ dW_t``

"""
function OrnsteinUhlenbeckProblem(a,r,σ,u0,tspan;kwargs...)
  f = function (u,p,t)
    a*(r-u)
  end
  g = function (u,p,t)
    σ
  end
  SDEProblem{false}(f,g,u0,tspan;kwargs...)
end


@doc doc"""

``dx = μ dt + σ dW_t``

"""
function GeometricBrownianMotionProblem(μ,σ,u0,tspan;kwargs...)
  f = function (u,p,t)
    μ
  end
  g = function (u,p,t)
    σ
  end
  SDEProblem{false}(f,g,u0,tspan;kwargs...)
end

@doc doc"""

``dx = σ(t)e^{at} dW_t``

"""
function MfStateProblem(a,σ,u0,tspan;kwargs...)
  f = function (u,p,t)
    0
  end
  g = function (u,p,t)
    σ(t)*exp(a*t)
  end
  SDEProblem{false}(f,g,u0,tspan;kwargs...)
end
