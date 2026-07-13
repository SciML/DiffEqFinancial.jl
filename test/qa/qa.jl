using SciMLTesting, DiffEqFinancial, Test
using JET

run_qa(
    DiffEqFinancial;
    explicit_imports = true,
    aqua_kwargs = (; deps_compat = (; check_extras = false)),
    api_docs_kwargs = (; rendered = true),
)
