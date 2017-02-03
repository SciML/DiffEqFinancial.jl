"""
dS = μSdt + \sqrt{v}SdW_1
dv = κ(Θ-v)dt + σ\sqrt{v}dW_2
dW_1 dW_2 = ρ dt
"""
type HestonProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  μ::tType
  κ::tType
  Θ::tType
  σ::tType
  ρ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function HestonProblem(μ,κ,Θ,σ,ρ,u0,tspan)
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
  HestonProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise).parameters[2],
              typeof(noise).parameters[1],
              typeof(f),typeof(g),
              typeof(noise).parameters[3]}(μ,κ,Θ,σ,ρ,u0,tspan,f,g,noise)
end

"""

``d \ln S(t) = (r(t) - q(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type GeneralizedBlackScholesProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3,thetaType,qType,rType} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  r::rType
  q::qType
  Θ::thetaType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function GeneralizedBlackScholesProblem(r,q,Θ,σ,u0,tspan)
  f = function (t,u)
    r(t) - q(t) - Θ(t,exp(u))^2 / 2
  end
  g = function (t,u)
    σ
  end
  noise = DiffEqBase.WHITE_NOISE
  isinplace = false
  GeneralizedBlackScholesProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise).parameters[2],
              typeof(noise).parameters[1],
              typeof(f),typeof(g),
              typeof(noise).parameters[3],
              typeof(Θ),typeof(q),typeof(r)}(r,q,Θ,σ,u0,tspan,f,g,noise)
end

"""

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type BlackScholesProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  r::tType
  Θ::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

BlackScholesProblem(r,Θ,σ,u0,tspan) = GeneralizedBlackScholesProblem(r,(t)->0,Θ,σ,u0,tspan)

"""

``dx = a(b(t)-x)dt + σ dW_t``

"""
type ExtendedOrnsteinUhlenbeckProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  a::tType
  b::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u0,tspan)
  f = function (t,u)
    a*(b(t)-u)
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  isinplace = false
  ExtendedOrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise).parameters[2],
              typeof(noise).parameters[1],
              typeof(f),typeof(g),
              typeof(noise).parameters[3]}(a,b,σ,u0,tspan,f,g,noise)
end

"""

``dx = a(r-x)dt + σ dW_t``

"""
type OrnsteinUhlenbeckProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  a::tType
  r::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function OrnsteinUhlenbeckProblem(a,r,σ,u0,tspan)
  f = function (t,u)
    a*(r-u)
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  isinplace = false
  OrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise).parameters[2],
              typeof(noise).parameters[1],
              typeof(f),typeof(g),
              typeof(noise).parameters[3]}(a,r,σ,u0,tspan,f,g,noise)
end


"""

``dx = μ dt + σ dW_t``

"""
type GeometricBrownianMotionProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  μ::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function OrnsteinUhlenbeckProblem(μ,σ,u0,tspan)
  f = function (t,u)
    μ
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  isinplace = false
  OrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise).parameters[2],
              typeof(noise).parameters[1],
              typeof(f),typeof(g),
              typeof(noise).parameters[3]}(a,r,σ,u0,tspan,f,g,noise)
end

"""

``dx = σ(t)e^{at} dW_t``

"""
type MfStateProblem{uType,tType,isinplace,isinplaceNoise,NoiseClass,F,F2,F3} <: AbstractSDEProblem{uType,tType,isinplace,NoiseClass,F,F2,F3}
  a::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NoiseProcess{NoiseClass,isinplaceNoise,F3}
end

function MfStateProblem(a,σ,u0,tspan)
  f = function (t,u)
    0
  end
  g = function (t,u)
    σ(t)*exp(a*t)
  end
  noise = WHITE_NOISE
  isinplace = false
  MfStateProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise).parameters[2],
              typeof(noise).parameters[1],
              typeof(f),typeof(g),
              typeof(noise).parameters[3]}(a,σ,u0,tspan,f,g,noise)
end
