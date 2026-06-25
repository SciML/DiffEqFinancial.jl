using SciMLTesting, DiffEqFinancial, Test
using JET

run_qa(
    DiffEqFinancial;
    explicit_imports = true,
    aqua_kwargs = (; deps_compat = (; check_extras = false)),
    ei_kwargs = (;
        # `AbstractSDEProblem` is owned by SciMLBase and reexported (non-public) by
        # DiffEqBase; it is not public in SciMLBase either, so neither the owner nor
        # the public import can be satisfied by re-pointing the import.
        all_explicit_imports_via_owners = (; ignore = (:AbstractSDEProblem,)),  # SciMLBase
        all_explicit_imports_are_public = (; ignore = (:AbstractSDEProblem,)),  # SciMLBase
    ),
)
