# Analytical Solutions

DiffEqFinancial.jl provides analytical (closed-form) solutions for the mean,
variance, and standard deviation of the supported financial stochastic processes.
These functions provide exact results without requiring Monte Carlo simulation,
offering significant performance and accuracy benefits.

## Geometric Brownian Motion

The GBM process: ``dX = \mu X \, dt + \sigma X \, dW_t``

```@docs
gbm_mean
gbm_variance
gbm_std
```

## Ornstein-Uhlenbeck Process

The OU process: ``dX = a(r - X) \, dt + \sigma \, dW_t``

```@docs
ou_mean
ou_variance
ou_std
ou_stationary_mean
ou_stationary_variance
```

## Cox-Ingersoll-Ross (CIR) Process

The CIR process: ``dr = \kappa(\theta - r) \, dt + \sigma \sqrt{r} \, dW_t``

```@docs
cir_mean
cir_variance
cir_std
cir_stationary_mean
cir_stationary_variance
```

## Black-Scholes Log-Price

The log-price process: ``d \ln S = (r - \sigma^2/2) \, dt + \sigma \, dW_t``

```@docs
bs_log_mean
bs_log_variance
bs_log_std
```

## Heston Model

The Heston stochastic volatility model:
``dS = \mu S \, dt + \sqrt{v} S \, dW_1``,
``dv = \kappa(\Theta - v) \, dt + \sigma \sqrt{v} \, dW_2``

```@docs
heston_mean
heston_variance_mean
heston_variance_variance
```
