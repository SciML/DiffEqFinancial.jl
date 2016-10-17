
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

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

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

function BlackScholesProblem(r,Θ,σ,u₀)
  f = function (t,u,du)
    r(t) - Θ(t,S)^2 / 2
  end
  g = function (t,u,du)
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
  HestonProblem(r,Θ,σ,u₀,f,g,analytic,knownanalytic,
                numvars,sizeu,isinplace,noise)
end
