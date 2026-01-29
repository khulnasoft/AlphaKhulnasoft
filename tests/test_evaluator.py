from alphakhulnasoft.evaluator import Evaluator


def test_evaluator_efficiency_score():
    evaluator = Evaluator()
    # Solved in 1 iteration
    score_1 = evaluator.calculate_efficiency_score(True, 1)
    assert score_1 == 1.0

    # Solved in 5 iterations
    score_5 = evaluator.calculate_efficiency_score(True, 5)
    assert score_5 == 0.2

    # Not solved
    score_fail = evaluator.calculate_efficiency_score(False, 5)
    assert score_fail == 0.0


def test_evaluator_add_result():
    evaluator = Evaluator()
    result = {"status": "SOLVED", "metrics": {"iterations": 2, "flow_depth": 1}}
    evaluator.add_result("Test Problem", result)
    assert len(evaluator.results) == 1
    assert evaluator.results[0]["solved"] is True
    assert evaluator.results[0]["iterations"] == 2
