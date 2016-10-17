"""
dS = μSdt + \sqrt{v}SdW_1
dv = κ(Θ-v)dt + σ\sqrt{v}dW_2
dW_1 dW_2 = ρ dt
"""
type HestonProblem <: AbstractSDEProblem
  μ
  κ
  Θ
  σ
  ρ
  u₀
  f
  g
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess
end

function HestonProblem(μ,κ,Θ,σ,ρ,u₀)
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
  knownanalytic = false
  analytic=(t,u,W)->0
  numvars = 2
  sizeu = (2,)
  if size(u₀) != sizeu
    err("Initial condtion must be a size 2 vector")
  end
  isinplace = true
  HestonProblem(μ,κ,Θ,σ,ρ,u₀,f,g,analytic,knownanalytic,
                numvars,sizeu,isinplace,noise)
end

"""

``d \ln S(t) = (r(t) - q(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type GeneralizedBlackScholesProblem
  r
  q
  Θ
  σ
  u₀
  f
  g
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess
end

function GeneralizedBlackScholesProblem(r,q,Θ,σ,u₀)
  f = function (t,u)
    r(t) - q(t) - Θ(t,exp(u))^2 / 2
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  knownanalytic = false
  analytic=(t,u,W)->0
  numvars = 1
  sizeu = (1,)
  if size(u₀) != sizeu
    err("Initial condtion must be a size 2 vector")
  end
  isinplace = false
  GeneralizedBlackScholesProblem(r,q,Θ,σ,u₀,f,g,analytic,knownanalytic,
                numvars,sizeu,isinplace,noise)
end

"""

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
type BlackScholesProblem
  r
  Θ
  σ
  u₀
  f
  g
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess
end

BlackScholesProblem(r,Θ,σ,u₀) = GeneralizedBlackScholesProblem(r,(t)->0,Θ,σ,u₀)

"""

``dx = a(b(t)-x)dt + σ dW_t``

"""
type ExtendedOrnsteinUhlenbeckProblem
  a
  b
  σ
  u₀
  f
  g
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess
end

function ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u₀)
  f = function (t,u)
    a*(b(t)-u)
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  knownanalytic = false
  analytic=(t,u,W)->0
  numvars = 1
  sizeu = (1,)
  if size(u₀) != sizeu
    err("Initial condtion must be a size 2 vector")
  end
  isinplace = false
  ExtendedOrnsteinUhlenbeckProblem(a,b,σ,u₀,f,g,analytic,knownanalytic,
                numvars,sizeu,isinplace,noise)
end

"""

``dx = a(r-x)dt + σ dW_t``

"""
type OrnsteinUhlenbeckProblem
  a
  r
  σ
  u₀
  f
  g
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess
end

function OrnsteinUhlenbeckProblem(a,r,σ,u₀)
  f = function (t,u)
    a*(r-u)
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  knownanalytic = false
  analytic=(t,u,W)->0
  numvars = 1
  sizeu = (1,)
  if size(u₀) != sizeu
    err("Initial condtion must be a size 2 vector")
  end
  isinplace = false
  OrnsteinUhlenbeckProblem(a,r,σ,u₀,f,g,analytic,knownanalytic,
                numvars,sizeu,isinplace,noise)
end


"""

``dx = μ dt + σ dW_t``

"""
type GeometricBrownianMotionProblem
  μ
  σ
  u₀
  f
  g
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess
end

function OrnsteinUhlenbeckProblem(μ,σ,u₀)
  f = function (t,u)
    μ
  end
  g = function (t,u)
    σ
  end
  noise = WHITE_NOISE
  knownanalytic = false
  analytic=(t,u,W)->0
  numvars = 1
  sizeu = (1,)
  if size(u₀) != sizeu
    err("Initial condtion must be a size 2 vector")
  end
  isinplace = false
  OrnsteinUhlenbeckProblem(a,r,σ,u₀,f,g,analytic,knownanalytic,
                numvars,sizeu,isinplace,noise)
end

"""

``dx = σ(t)e^{at} dW_t``

"""
type MfStateProblem
  a
  σ
  u₀
  f
  g
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  sizeu#::Tuple
  isinplace::Bool
  noise::NoiseProcess
end

function MfStateProblem(a,σ,u₀)
  f = function (t,u)
    0
  end
  g = function (t,u)
    σ(t)*exp(a*t)
  end
  noise = WHITE_NOISE
  knownanalytic = false
  analytic=(t,u,W)->0
  numvars = 1
  sizeu = (1,)
  if size(u₀) != sizeu
    err("Initial condtion must be a size 2 vector")
  end
  isinplace = false
  MfStateProblem(a,σ,u₀,f,g,analytic,knownanalytic,
                numvars,sizeu,isinplace,noise)
end
