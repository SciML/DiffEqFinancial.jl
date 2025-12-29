# Analytical solutions, mean, and variance calculations for financial models

# =============================================================================
# Geometric Brownian Motion: dx = μx dt + σx dW_t
# =============================================================================

@doc doc"""
Analytical mean of Geometric Brownian Motion.

```math
E[X(t)] = x_0 e^{\mu t}
```

### Arguments
- `μ`: Drift coefficient
- `u0`: Initial value ``x_0``
- `t`: Time (scalar or array)

### Returns
The expected value at time `t`.
"""
function gbm_mean(μ, u0, t)
    return u0 * exp(μ * t)
end

@doc doc"""
Analytical variance of Geometric Brownian Motion.

```math
\text{Var}[X(t)] = x_0^2 e^{2\mu t} (e^{\sigma^2 t} - 1)
```

### Arguments
- `μ`: Drift coefficient
- `σ`: Volatility coefficient
- `u0`: Initial value ``x_0``
- `t`: Time (scalar or array)

### Returns
The variance at time `t`.
"""
function gbm_variance(μ, σ, u0, t)
    return u0^2 * exp(2μ * t) * expm1(σ^2 * t)
end

@doc doc"""
Analytical standard deviation of Geometric Brownian Motion.

### Arguments
- `μ`: Drift coefficient
- `σ`: Volatility coefficient
- `u0`: Initial value ``x_0``
- `t`: Time (scalar or array)

### Returns
The standard deviation at time `t`.
"""
function gbm_std(μ, σ, u0, t)
    return sqrt(gbm_variance(μ, σ, u0, t))
end

# =============================================================================
# Ornstein-Uhlenbeck Process: dx = a(r - x)dt + σ dW_t
# =============================================================================

@doc doc"""
Analytical mean of the Ornstein-Uhlenbeck process.

```math
E[X(t)] = r + (x_0 - r) e^{-at}
```

### Arguments
- `a`: Mean reversion speed
- `r`: Long-run mean level
- `u0`: Initial value ``x_0``
- `t`: Time (scalar or array)

### Returns
The expected value at time `t`.
"""
function ou_mean(a, r, u0, t)
    return r + (u0 - r) * exp(-a * t)
end

@doc doc"""
Analytical variance of the Ornstein-Uhlenbeck process.

```math
\text{Var}[X(t)] = \frac{\sigma^2}{2a} (1 - e^{-2at})
```

### Arguments
- `a`: Mean reversion speed
- `σ`: Volatility coefficient
- `t`: Time (scalar or array)

### Returns
The variance at time `t`.
"""
function ou_variance(a, σ, t)
    return (σ^2 / (2a)) * (-expm1(-2a * t))
end

@doc doc"""
Analytical standard deviation of the Ornstein-Uhlenbeck process.

### Arguments
- `a`: Mean reversion speed
- `σ`: Volatility coefficient
- `t`: Time (scalar or array)

### Returns
The standard deviation at time `t`.
"""
function ou_std(a, σ, t)
    return sqrt(ou_variance(a, σ, t))
end

@doc doc"""
Stationary (long-run) mean of the Ornstein-Uhlenbeck process.

```math
\lim_{t \to \infty} E[X(t)] = r
```

### Arguments
- `r`: Long-run mean level

### Returns
The stationary mean.
"""
function ou_stationary_mean(r)
    return r
end

@doc doc"""
Stationary (long-run) variance of the Ornstein-Uhlenbeck process.

```math
\lim_{t \to \infty} \text{Var}[X(t)] = \frac{\sigma^2}{2a}
```

### Arguments
- `a`: Mean reversion speed
- `σ`: Volatility coefficient

### Returns
The stationary variance.
"""
function ou_stationary_variance(a, σ)
    return σ^2 / (2a)
end

# =============================================================================
# Cox-Ingersoll-Ross (CIR) Process: dr = κ(θ - r)dt + σ√r dW_t
# =============================================================================

@doc doc"""
Analytical mean of the CIR process.

```math
E[r(t)] = \theta + (r_0 - \theta) e^{-\kappa t}
```

### Arguments
- `κ`: Mean reversion speed
- `θ`: Long-run mean level
- `u0`: Initial value ``r_0``
- `t`: Time (scalar or array)

### Returns
The expected value at time `t`.
"""
function cir_mean(κ, θ, u0, t)
    return θ + (u0 - θ) * exp(-κ * t)
end

@doc doc"""
Analytical variance of the CIR process.

```math
\text{Var}[r(t)] = r_0 \frac{\sigma^2}{\kappa} (e^{-\kappa t} - e^{-2\kappa t}) + \frac{\theta \sigma^2}{2\kappa} (1 - e^{-\kappa t})^2
```

### Arguments
- `κ`: Mean reversion speed
- `θ`: Long-run mean level
- `σ`: Volatility coefficient
- `u0`: Initial value ``r_0``
- `t`: Time (scalar or array)

### Returns
The variance at time `t`.
"""
function cir_variance(κ, θ, σ, u0, t)
    exp_κt = exp(-κ * t)
    exp_2κt = exp(-2κ * t)
    term1 = u0 * (σ^2 / κ) * (exp_κt - exp_2κt)
    term2 = (θ * σ^2 / (2κ)) * (1 - exp_κt)^2
    return term1 + term2
end

@doc doc"""
Analytical standard deviation of the CIR process.

### Arguments
- `κ`: Mean reversion speed
- `θ`: Long-run mean level
- `σ`: Volatility coefficient
- `u0`: Initial value ``r_0``
- `t`: Time (scalar or array)

### Returns
The standard deviation at time `t`.
"""
function cir_std(κ, θ, σ, u0, t)
    return sqrt(cir_variance(κ, θ, σ, u0, t))
end

@doc doc"""
Stationary (long-run) mean of the CIR process.

```math
\lim_{t \to \infty} E[r(t)] = \theta
```

### Arguments
- `θ`: Long-run mean level

### Returns
The stationary mean.
"""
function cir_stationary_mean(θ)
    return θ
end

@doc doc"""
Stationary (long-run) variance of the CIR process.

```math
\lim_{t \to \infty} \text{Var}[r(t)] = \frac{\theta \sigma^2}{2\kappa}
```

### Arguments
- `κ`: Mean reversion speed
- `θ`: Long-run mean level
- `σ`: Volatility coefficient

### Returns
The stationary variance.
"""
function cir_stationary_variance(κ, θ, σ)
    return θ * σ^2 / (2κ)
end

# =============================================================================
# Black-Scholes Log-Price: d ln S = (r - σ²/2)dt + σ dW_t
# =============================================================================

@doc doc"""
Analytical mean of the Black-Scholes log-price process.

```math
E[\ln S(t)] = \ln S_0 + (r - \frac{\sigma^2}{2}) t
```

### Arguments
- `r`: Risk-free rate (constant)
- `σ`: Volatility coefficient
- `u0`: Initial log-price ``\ln S_0``
- `t`: Time (scalar or array)

### Returns
The expected log-price at time `t`.
"""
function bs_log_mean(r, σ, u0, t)
    return u0 + (r - σ^2 / 2) * t
end

@doc doc"""
Analytical variance of the Black-Scholes log-price process.

```math
\text{Var}[\ln S(t)] = \sigma^2 t
```

### Arguments
- `σ`: Volatility coefficient
- `t`: Time (scalar or array)

### Returns
The variance of log-price at time `t`.
"""
function bs_log_variance(σ, t)
    return σ^2 * t
end

@doc doc"""
Analytical standard deviation of the Black-Scholes log-price process.

### Arguments
- `σ`: Volatility coefficient
- `t`: Time (scalar or array)

### Returns
The standard deviation of log-price at time `t`.
"""
function bs_log_std(σ, t)
    return σ * sqrt(t)
end

# =============================================================================
# Heston Model: dS = μS dt + √v S dW₁, dv = κ(Θ-v)dt + σ√v dW₂
# =============================================================================

@doc doc"""
Analytical mean of the Heston model asset price.

```math
E[S(t)] = S_0 e^{\mu t}
```

Note: The mean is independent of the stochastic volatility path because
the volatility term has zero expected increment.

### Arguments
- `μ`: Drift coefficient
- `u0`: Initial asset price ``S_0`` (first component of initial state)
- `t`: Time (scalar or array)

### Returns
The expected asset price at time `t`.
"""
function heston_mean(μ, u0, t)
    return u0 * exp(μ * t)
end

@doc doc"""
Analytical mean of the Heston model variance process.

The variance process follows CIR dynamics: ``dv = \kappa(\Theta - v)dt + \sigma\sqrt{v}dW_2``

```math
E[v(t)] = \Theta + (v_0 - \Theta) e^{-\kappa t}
```

### Arguments
- `κ`: Mean reversion speed
- `Θ`: Long-run variance level
- `v0`: Initial variance ``v_0`` (second component of initial state)
- `t`: Time (scalar or array)

### Returns
The expected variance at time `t`.
"""
function heston_variance_mean(κ, Θ, v0, t)
    return cir_mean(κ, Θ, v0, t)
end

@doc doc"""
Analytical variance of the Heston model variance process.

The variance process follows CIR dynamics.

### Arguments
- `κ`: Mean reversion speed
- `Θ`: Long-run variance level
- `σ`: Vol-of-vol coefficient
- `v0`: Initial variance ``v_0``
- `t`: Time (scalar or array)

### Returns
The variance of the variance process at time `t`.
"""
function heston_variance_variance(κ, Θ, σ, v0, t)
    return cir_variance(κ, Θ, σ, v0, t)
end
