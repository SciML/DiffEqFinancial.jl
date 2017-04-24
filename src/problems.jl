"""
dS = μSdt + \sqrt{v}SdW_1
dv = κ(Θ-v)dt + σ\sqrt{v}dW_2
dW_1 dW_2 = ρ dt
"""
type HestonProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  μ::tType
  κ::tType
  Θ::tType
  σ::tType
  ρ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
end

function HestonProblem(μ,κ,Θ,σ,ρ,u0,tspan;callback = CallbackSet())
  f = function (t,u,du)
    du[1] = μ*u[1]
    du[2] = κ*(Θ-u[2])
  end
  g = function (t,u,du)
    du[1] = √u[2]*u[1]
    du[2] = Θ*√u[2]
  end
  Γ = [1 ρ;ρ 1] # Covariance Matrix
  mass_matrix=I
  noise_rate_prototype = nothing
  noise = CorrelatedWienerProcess!(Γ,tspan[1],zeros(2),zeros(2))
  isinplace = true
  HestonProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),
              typeof(mass_matrix)}(μ,κ,Θ,σ,ρ,u0,tspan,f,g,noise,callback,noise_rate_prototype,mass_matrix)
end

"""

``d \ln S(t) = (r(t) - q(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type GeneralizedBlackScholesProblem{uType,tType,isinplace,NP,F,F2,thetaType,qType,rType,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  r::rType
  q::qType
  Θ::thetaType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
end

function GeneralizedBlackScholesProblem(r,q,Θ,σ,u0,tspan;callback = CallbackSet())
  f = function (t,u)
    r(t) - q(t) - Θ(t,exp(u))^2 / 2
  end
  g = function (t,u)
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
              typeof(callback),typeof(noise_rate_prototype),typeof(mass_matrix)}(r,q,Θ,σ,u0,tspan,f,g,noise,callback,noise_rate_prototype,mass_matrix)
end

"""

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type BlackScholesProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  r::tType
  Θ::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
end

BlackScholesProblem(r,Θ,σ,u0,tspan;callback = CallbackSet(),
                    noise_rate_prototype = nothing) =
                    GeneralizedBlackScholesProblem(r,(t)->0,Θ,σ,u0,tspan,callback=callback)

"""

``dx = a(b(t)-x)dt + σ dW_t``

"""
type ExtendedOrnsteinUhlenbeckProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  a::tType
  b::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
end

function ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u0,tspan;callback = CallbackSet())
  f = function (t,u)
    a*(b(t)-u)
  end
  g = function (t,u)
    σ
  end
  noise_rate_prototype = nothing
  noise = nothing
  isinplace = false
  mass_matrix=I
  ExtendedOrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),typeof(mass_matrix)}(a,b,σ,u0,tspan,f,g,noise,callback,noise_rate_prototype,mass_matrix)
end

"""

``dx = a(r-x)dt + σ dW_t``

"""
type OrnsteinUhlenbeckProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  a::tType
  r::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
end

function OrnsteinUhlenbeckProblem(a,r,σ,u0,tspan;callback = CallbackSet())
  f = function (t,u)
    a*(r-u)
  end
  g = function (t,u)
    σ
  end
  noise = nothing
  isinplace = false
  noise_rate_prototype = nothing
  mass_matrix=I
  OrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),typeof(mass_matrix)}(a,r,σ,u0,tspan,f,g,noise,callback,noise_rate_prototype,mass_matrix)
end


"""

``dx = μ dt + σ dW_t``

"""
type GeometricBrownianMotionProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  μ::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
end

function OrnsteinUhlenbeckProblem(μ,σ,u0,tspan;callback = CallbackSet())
  f = function (t,u)
    μ
  end
  g = function (t,u)
    σ
  end
  noise = nothing
  isinplace = false
  mass_matrix=I
  noise_rate_prototype = nothing
  OrnsteinUhlenbeckProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),typeof(mass_matrix)}(a,r,σ,u0,tspan,f,g,noise,callback,noise_rate_prototype,mass_matrix)
end

"""

``dx = σ(t)e^{at} dW_t``

"""
type MfStateProblem{uType,tType,isinplace,NP,F,F2,C,ND,MM} <: AbstractSDEProblem{uType,tType,isinplace,ND}
  a::tType
  σ::tType
  u0::uType
  tspan::Tuple{tType,tType}
  f::F
  g::F2
  noise::NP
  callback::C
  noise_rate_prototype::ND
  mass_matrix::MM
end

function MfStateProblem(a,σ,u0,tspan;callback = CallbackSet(),noise_rate_prototype = nothing)
  f = function (t,u)
    0
  end
  g = function (t,u)
    σ(t)*exp(a*t)
  end
  noise = nothing
  isinplace = false
  mass_matrix=I
  MfStateProblem{typeof(u0),eltype(tspan),isinplace,
              typeof(noise),
              typeof(f),typeof(g),
              typeof(callback),typeof(noise_rate_prototype),typeof(mass_matrix)}(
              a,σ,u0,tspan,f,g,noise,callback,noise_rate_prototype,mass_matrix)
end
