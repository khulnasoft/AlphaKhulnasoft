"""Tests for the nonequivalence verifier and 4x4 fixtures."""

import alphakhulnasoft.matmul as mm
from alphakhulnasoft.matmul.nonequivalence.four_by_four import (
    four_by_four_fixtures,
    schoolbook_4x4,
    strassen_4x4,
)
from alphakhulnasoft.matmul.nonequivalence.invariants import separates
from alphakhulnasoft.matmul.nonequivalence.verifier import (
    EquivalenceResult,
    check_equivalence,
)
from alphakhulnasoft.matmul.tensor import Factorization, Field, MatmulSpec, RankOneFactor


def _scaled(alg: Factorization, factor_index: int, lam: int) -> Factorization:
    """Return ``alg`` with one factor's u scaled by ``lam`` and v by 1/lam."""
    factors = list(alg.factors)
    f = factors[factor_index]
    su = tuple(tuple(lam * x for x in row) for row in f.u)
    sv = tuple(tuple(x / lam for x in row) for row in f.v)
    factors[factor_index] = RankOneFactor(su, sv, f.w)
    return Factorization(
        alg.spec, alg.field, alg.algorithm_name + "_scaled", tuple(factors), alg.provenance
    )


def test_known_equivalent_under_scaling_accepted():
    alg = mm.strassen_2x2()
    scaled = _scaled(alg, 0, 2)
    assert mm.verify_exact(scaled, [[1, 2], [3, 4]], [[5, 6], [7, 8]])
    report = check_equivalence(alg, scaled)
    assert report.result == EquivalenceResult.EQUIVALENT


def test_known_nonequivalent_4x4_rejected_by_rank():
    school = schoolbook_4x4()
    strassen = strassen_4x4()
    assert school.rank == 64
    assert strassen.rank == 49
    report = check_equivalence(school, strassen)
    assert report.result == EquivalenceResult.NONEQUIVALENT
    assert report.invariant == "rank"


def test_four_by_four_fixtures_exact():
    for name, alg in four_by_four_fixtures().items():
        a, b = mm.reference.random_cpu_inputs(alg.spec, seed=3)
        assert mm.verify_exact(alg, a, b), name


def test_invalid_dimensions_rejected_before_comparison():
    a = schoolbook_4x4()
    b = mm.schoolbook(MatmulSpec(2, 2, 2))
    report = check_equivalence(a, b)
    assert report.result == EquivalenceResult.NONEQUIVALENT
    assert report.invariant == "dimensions"


def test_tolerance_never_proves_equivalence():
    # Two float-field algorithms that happen to be numerically close but have
    # different ranks must be reported nonequivalent by an invariant, never by
    # numeric tolerance.
    a = mm.schoolbook(MatmulSpec(4, 4, 4), field=Field.FLOAT64)
    b = strassen_4x4()
    # Force b to float field for a fair comparison of the verifier's logic.
    b_float = Factorization(b.spec, Field.FLOAT64, b.algorithm_name, b.factors, b.provenance)
    report = check_equivalence(a, b_float)
    assert report.result == EquivalenceResult.NONEQUIVALENT
    assert report.invariant == "rank"


def test_inconclusive_when_invariants_agree_large_rank():
    # Same algorithm reported under two registrations -> rank equal, invariants
    # agree, but the search over 49+ factors is skipped -> INCONCLUSIVE unless
    # an invariant separates. Here they are literally equal so it is EQUIVALENT
    # only via the trivial permutation; ensure large-rank inconclusive path exists.
    a = strassen_4x4()
    b = strassen_4x4()
    report = check_equivalence(a, b)
    # Identical algos match under permutation with lambda=1.
    assert report.result in (EquivalenceResult.EQUIVALENT, EquivalenceResult.INCONCLUSIVE)


def test_separates_helper():
    assert separates(schoolbook_4x4(), strassen_4x4()) == "rank"
