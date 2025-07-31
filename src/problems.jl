@doc doc"""

```math
dS = μSdt + \sqrt{v}SdW_1
dv = κ(Θ-v)dt + σ\sqrt{v}dW_2
dW_1 dW_2 = ρ dt
```

### Keyword Arguments
- `modifier`: A function applied inside the square root in the diffusion term. By default, modifier ensures numerical stability when `u[2]` becomes slightly negative due to discretization errors. Without this, the square root of a negative number would result in a domain error. You may override this if using an alternative regularization strategy or if you're certain `u[2]` will remain positive.

"""
function HestonProblem(
        μ, κ, Θ, σ, ρ, u0, tspan; seed = UInt64(0), modifier = x -> max(x, 0), kwargs...)
    f = function (du, u, p, t)
        du[1] = μ * u[1]
        du[2] = κ * (Θ - modifier(u[2]))
    end
    g = function (du, u, p, t)
        du[1] = √modifier(u[2]) * u[1]
        du[2] = σ * √modifier(u[2])
    end
    Γ = [1 ρ; ρ 1] # Covariance Matrix
    noise_rate_prototype = nothing

    if seed == 0
        seed = rand(UInt64)
    end
    noise = CorrelatedWienerProcess!(Γ, tspan[1], zeros(2), zeros(2),
        rng = Xorshifts.Xoroshiro128Plus(seed))

    sde_f = SDEFunction{true}(f, g)
    SDEProblem(sde_f, u0, tspan, noise = noise, seed = seed, kwargs...)
end

@doc doc"""

``d \ln S(t) = (r(t) - q(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
function GeneralizedBlackScholesProblem(r, q, Θ, σ, u0, tspan; kwargs...)
    f = function (u, p, t)
        r(t) - q(t) - Θ(t, exp(u))^2 / 2
    end
    g = function (u, p, t)
        σ
    end
    SDEProblem{false}(f, g, u0, tspan; kwargs...)
end

@doc doc"""

``d \ln S(t) = (r(t) - \frac{Θ(t,S)^2}{2})dt + σ dW_t``

Solves for ``log S(t)``.
"""
mutable struct BlackScholesProblem{uType, tType, tupType, isinplace, NP, F, F2, C, ND, MM
} <:
               DiffEqBase.AbstractSDEProblem{uType, tType, isinplace, ND}
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

function BlackScholesProblem(r, Θ, σ, u0, tspan; callback = CallbackSet(),
        noise_rate_prototype = nothing, seed = UInt64(0))
    GeneralizedBlackScholesProblem(r, (t) -> 0, Θ, σ, u0, tspan, callback = callback,
        seed = seed)
end

@doc doc"""

``dx = a(b(t)-x)dt + σ dW_t``

"""
function ExtendedOrnsteinUhlenbeckProblem(a, b, σ, u0, tspan; kwargs...)
    f = function (u, p, t)
        a * (b(t) - u)
    end
    g = function (u, p, t)
        σ
    end
    SDEProblem{false}(f, g, u0, tspan; kwargs...)
end

@doc doc"""

``dx = a(r-x)dt + σ dW_t``

"""
function OrnsteinUhlenbeckProblem(a, r, σ, u0, tspan; kwargs...)
    f = function (u, p, t)
        a * (r - u)
    end
    g = function (u, p, t)
        σ
    end
    SDEProblem{false}(f, g, u0, tspan; kwargs...)
end

@doc doc"""

``dx = μx dt + σx dW_t``

"""
function GeometricBrownianMotionProblem(μ, σ, u0, tspan; kwargs...)
    f = function (u, p, t)
        μ * u
    end
    g = function (u, p, t)
        σ * u
    end
    SDEProblem{false}(f, g, u0, tspan; kwargs...)
end

@doc doc"""

``dx = σ(t)e^{at} dW_t``

"""
function MfStateProblem(a, σ, u0, tspan; kwargs...)
    f = function (u, p, t)
        0
    end
    g = function (u, p, t)
        σ(t) * exp(a * t)
    end
    SDEProblem{false}(f, g, u0, tspan; kwargs...)
end

@doc doc"""

``dr = κ(θ - r)dt + σ√r dW_t``

The Cox-Ingersoll-Ross (CIR) model is commonly used for short-rate modeling in interest rate theory.

### Keyword Arguments
- `modifier`: A function applied inside the square root in the diffusion term. It ensures rate positivity which can break due to discretization error.

"""
function CIRProblem(κ, θ, σ, u0, tspan; modifier = x -> max(x, 0), kwargs...)
    if 2κ * θ < σ^2
        @warn "Feller condition 2κθ ≥ σ² is violated. The CIR process may reach zero."
    end

    f = function (u, p, t)
        κ * (θ - modifier(u))
    end
    g = function (u, p, t)
        σ * sqrt(modifier(u))
    end
    SDEProblem{false}(f, g, u0, tspan; kwargs...)
end

@doc doc"""

``dr = \kappa(\theta - r)dt + \sigma\sqrt{r} dW_t``

The Cox-Ingersoll-Ross (CIR) model is commonly used for short-rate modeling in interest rate theory.

This type represents the exact transition law of the CIR process, and can be used to sample directly from the known non-central \(\chi^2\) distribution implied by the model.

### Fields
- `κ `: Mean-reversion speed.
- `θ`: Long-run mean level.
- `σ`: Volatility coefficient.

This exact law is used internally by `CIRNoise` to create a `NoiseProcess` with the correct distributional dynamics.

"""
struct CoxIngersollRoss{T1, T2, T3}
    κ::T1
    θ::T2
    σ::T3
end

@doc doc"""
Samples the CIR process exactly using the non-central chi-squared transition distribution.

### Arguments
- `DW`: Not used but required for interface compatibility.
- `W`: Path history; last value is used for adjustment.
- `dt`: Time step size.
- `u`: Current value (not used in exact sampling).
- `p`: Parameters (not used here).
- `t`: Current time (not used here).
- `rng`: Random number generator.

Returns an increment from the single sample from the exact transition distribution.

"""
function (X::CoxIngersollRoss)(DW, W, dt, u, p, t, rng) #dist
    κ, θ, σ = X.κ, X.θ, X.σ
    d = 4 * κ * θ / σ^2  # Degrees of freedom
    λ = -4 * κ * exp(-κ * dt) * W.W[end] / (σ^2 * expm1(-κ * dt))  # Noncentrality parameter
    c = -σ^2 * expm1(-κ * dt) / 4κ  # Scaling factor
    sample = c * Distributions.rand(rng, NoncentralChisq(d, λ))
    return sample - W.W[end] #return the increment
end

@doc doc"""

``dr = κ(θ - r)dt + σ√r dW_t``

The Cox-Ingersoll-Ross (CIR) model is commonly used for short-rate modeling in interest rate theory.
This is a distributionally-exact process, leveraging the known χ² transition law of the process.
The sampling leverages Distributions.jl.
This method is way slower than the discretized version, hence it is advisable to use it only when simulation at few time-points is needed with no bias.
"""
function CIRNoise(κ, θ, σ, t0, W0, Z0 = nothing; kwargs...)
    cir = CoxIngersollRoss(κ, θ, σ)
    return NoiseProcess{false}(t0, W0, Z0, cir, nothing)
end
