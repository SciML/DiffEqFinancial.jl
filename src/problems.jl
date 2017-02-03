"""
dS = μSdt + \sqrt{v}SdW_1
dv = κ(Θ-v)dt + σ\sqrt{v}dW_2
dW_1 dW_2 = ρ dt
"""
type HestonProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  μ::uType
  κ::uType
  Θ::uType
  σ::uType
  ρ::uType
  u₀::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  isinplace::Bool
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function HestonProblem(μ,κ,Θ,σ,ρ,u₀,tspan)
  f = function (t,u,du)
    du[1] = μ*u[1]
    du[2] = κ*(Θ-u[2])
  end
  g = function (t,u,du)
    du[1] = √u[2]*u[1]
    du[2] = Θ*√u[2]
  end
  Γ = [1 ρ;ρ 1] # Covariance Matrix
  noise = construct_correlated_noisefunc(Γ)
  isinplace = true
  HestonProblem(μ,κ,Θ,σ,ρ,u₀,tspan,f,g,isinplace,noise)
end

"""

``d \ln S(t) = (r(t) - q(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type GeneralizedBlackScholesProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  r::uType
  q::uType
  Θ::uType
  σ::uType
  u₀::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  isinplace::Bool
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function GeneralizedBlackScholesProblem(r,q,Θ,σ,u₀,tspan)
  f = function (t,u)
    r(t) - q(t) - Θ(t,exp(u))^2 / 2
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  isinplace = false
  GeneralizedBlackScholesProblem(r,q,Θ,σ,u₀,tspan,f,g,isinplace,noise)
end

"""

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type BlackScholesProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  r::uType
  Θ::uType
  σ::uType
  u₀::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  isinplace::Bool
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

BlackScholesProblem(r,Θ,σ,u₀,tspan) = GeneralizedBlackScholesProblem(r,(t)->0,Θ,σ,u₀,tspan)

"""

``dx = a(b(t)-x)dt + σ dW_t``

"""
type ExtendedOrnsteinUhlenbeckProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  a::uType
  b::uType
  σ::uType
  u₀::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u₀,tspan)
  f = function (t,u)
    a*(b(t)-u)
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  isinplace = false
  ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u₀,tspan,f,g,isinplace,noise)
end

"""

``dx = a(r-x)dt + σ dW_t``

"""
type OrnsteinUhlenbeckProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  a::uType
  r::uType
  σ::uType
  u₀::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  isinplace::Bool
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function OrnsteinUhlenbeckProblem(a,r,σ,u₀,tspan)
  f = function (t,u)
    a*(r-u)
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  isinplace = false
  OrnsteinUhlenbeckProblem(a,r,σ,u₀,tspan,f,g,isinplace,noise)
end


"""

``dx = μ dt + σ dW_t``

"""
type GeometricBrownianMotionProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  μ::uType
  σ::uType
  u₀::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  isinplace::Bool
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function OrnsteinUhlenbeckProblem(μ,σ,u₀,tspan)
  f = function (t,u)
    μ
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  isinplace = false
  OrnsteinUhlenbeckProblem(a,r,σ,u₀,tspan,f,g,isinplace,noise)
end

"""

``dx = σ(t)e^{at} dW_t``

"""
type MfStateProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  a::uType
  σ::uType
  u₀::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  isinplace::Bool
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function MfStateProblem(a,σ,u₀,tspan)
  f = function (t,u)
    0
  end
  g = function (t,u)
    σ(t)*exp(a*t)
  end
  noise = WHITE_NOISE
  isinplace = false
  MfStateProblem(a,σ,u₀,tspan,f,g,isinplace,noise)
end
