"""
dS = μSdt + \sqrt{v}SdW_1
dv = κ(Θ-v)dt + σ\sqrt{v}dW_2
dW_1 dW_2 = ρ dt
"""
mutable struct HestonProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  μ::tType
  κ::tType
  Θ::tType
  σ::tType
  ρ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  p::Void
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
  seed::UInt64
end

function HestonProblem(μ,κ,Θ,σ,ρ,u0,tspan;callback = CallbackSet(),seed=UInt64(0))
  f = function (du,u,p,t)
    du[1] = μ*u[1]
    du[2] = κ*(Θ-u[2])
  end
  g = function (du,u,p,t)
    du[1] = √u[2]*u[1]
    du[2] = Θ*√u[2]
  end
  Γ = [1 ρ;ρ 1] # Covariance Matrix
  mass_matrix=I
  noise_rate_prototype = nothing
  if seed == 0
    seed = rand(UInt64)
  end
  noise = CorrelatedWienerProcess!(Γ,tspan[1],zeros(2),zeros(2),rng = Xorshifts.Xoroshiro128Plus(seed))
  isinplace = true
  HestonProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),
              typeof(mass_matrix)}(μ,κ,Θ,σ,ρ,u0,tspan,nothing,
              f,g,noise,callback,noise_rate_prototype,mass_matrix,seed)
end

"""

``d \ln S(t) = (r(t) - q(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
mutable struct GeneralizedBlackScholesProblem{uType,tType,isinplace,NP,F,F2,thetaType,qType,rType,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  r::rType
  q::qType
  Θ::thetaType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  p::Void
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
  seed::UInt64
end

function GeneralizedBlackScholesProblem(r,q,Θ,σ,u0,tspan;callback = CallbackSet(),seed=UInt64(0))
  f = function (u,p,t)
    r(t) - q(t) - Θ(t,exp(u))^2 / 2
  end
  g = function (u,p,t)
    σ
  end
  noise_rate_prototype = nothing
  noise = nothing
  isinplace = false
  mass_matrix=I
  GeneralizedBlackScholesProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(Θ),typeof(q),typeof(r),
              typeof(callback),typeof(noise_rate_prototype),typeof(mass_matrix)}(
              r,q,Θ,σ,u0,tspan,nothing,f,g,noise,callback,
              noise_rate_prototype,mass_matrix,seed)
end

"""

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
mutable struct BlackScholesProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  r::tType
  Θ::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  p::Void
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

"""

``dx = a(b(t)-x)dt + σ dW_t``

"""
mutable struct ExtendedOrnsteinUhlenbeckProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  a::tType
  b::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  p::Void
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
  seed::UInt64
end

function ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u0,tspan;callback = CallbackSet(),seed=UInt64(0))
  f = function (u,p,t)
    a*(b(t)-u)
  end
  g = function (u,p,t)
    σ
  end
  noise_rate_prototype = nothing
  noise = nothing
  isinplace = false
  mass_matrix=I
  ExtendedOrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),
              typeof(mass_matrix)}(a,b,σ,u0,tspan,p,f,g,noise,
              callback,noise_rate_prototype,mass_matrix,seed)
end

"""

``dx = a(r-x)dt + σ dW_t``

"""
mutable struct OrnsteinUhlenbeckProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  a::tType
  r::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  p::Void
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
  seed::UInt64
end

function OrnsteinUhlenbeckProblem(a,r,σ,u0,tspan;callback = CallbackSet(),seed=UInt64(0))
  f = function (u,p,t)
    a*(r-u)
  end
  g = function (u,p,t)
    σ
  end
  noise = nothing
  isinplace = false
  noise_rate_prototype = nothing
  mass_matrix=I
  OrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),
              typeof(mass_matrix)}(a,r,σ,u0,tspan,p,f,g,noise,callback,
              noise_rate_prototype,mass_matrix,seed)
end


"""

``dx = μ dt + σ dW_t``

"""
mutable struct GeometricBrownianMotionProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  μ::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  p::Void
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
  seed::UInt64
end

function OrnsteinUhlenbeckProblem(μ,σ,u0,tspan;callback = CallbackSet(),seed=UInt64(0))
  f = function (u,p,t)
    μ
  end
  g = function (u,p,t)
    σ
  end
  noise = nothing
  isinplace = false
  mass_matrix=I
  noise_rate_prototype = nothing
  OrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),
              typeof(mass_matrix)}(a,r,σ,u0,tspan,nothing,f,g,noise,callback,
              noise_rate_prototype,mass_matrix,seed)
end

"""

``dx = σ(t)e^{at} dW_t``

"""
mutable struct MfStateProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  a::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  p::Void
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
  seed::UInt64
end

function MfStateProblem(a,σ,u0,tspan;callback = CallbackSet(),noise_rate_prototype = nothing,seed=UInt64(0))
  f = function (u,p,t)
    0
  end
  g = function (u,p,t)
    σ(t)*exp(a*t)
  end
  noise = nothing
  isinplace = false
  mass_matrix=I
  MfStateProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),typeof(mass_matrix)}(
              a,σ,u0,tspan,nothing,f,g,noise,callback,
              noise_rate_prototype,mass_matrix,seed)
end
