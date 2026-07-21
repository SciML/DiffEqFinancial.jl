using SciMLTesting, DiffEqFinancial, Test
using JET

run_qa(
    DiffEqFinancial;
    aqua_kwargs = (; deps_compat = (; check_extras = false)),
)
