@setup_workload begin
    precompile_heston = HestonProblem(
        0.03, 1.2, 0.2, 0.5, 0.25, [100.0, 0.04], (0.0, 1.0);
        seed = UInt64(1)
    )
    precompile_gbm = GeometricBrownianMotionProblem(
        0.03, 0.2, 100.0, (0.0, 1.0)
    )

    @compile_workload begin
        precompile_heston.f.f(
            similar(precompile_heston.u0),
            precompile_heston.u0,
            precompile_heston.p,
            0.5
        )
        precompile_heston.f.g(
            similar(precompile_heston.u0),
            precompile_heston.u0,
            precompile_heston.p,
            0.5
        )
        precompile_gbm.f(precompile_gbm.u0, precompile_gbm.p, 0.5)
        precompile_gbm.g(precompile_gbm.u0, precompile_gbm.p, 0.5)
        gbm_mean(0.03, 100.0, 0.5)
        gbm_variance(0.03, 0.2, 100.0, 0.5)
        ou_mean(0.5, 1.0, 0.0, 0.5)
        cir_variance(0.5, 0.04, 0.1, 0.03, 0.5)
        heston_variance_mean(1.2, 0.04, 0.04, 0.5)
    end
end
