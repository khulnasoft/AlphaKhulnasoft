from alphakhulnasoft.data_loader import DataLoader


def test_get_mock_problem():
    loader = DataLoader()
    problem = loader.get_mock_problem()
    assert problem["id"] == "mock-001"
    assert problem["title"] == "Double and Positives"
    assert len(problem["tests"]) == 3
    assert problem["tests"][0] == {"input": "2", "expected": "4"}
    assert problem["tests"][1] == {"input": "-5", "expected": "0"}
    assert problem["tests"][2] == {"input": "0", "expected": "0"}


def test_load_problems_missing_file():
    loader = DataLoader()
    problems = loader.load_problems("/nonexistent/path.jsonl")
    assert problems == []


def test_load_problems_empty_path():
    loader = DataLoader()
    problems = loader.load_problems("")
    assert problems == []


def test_data_loader_default_dataset_name():
    loader = DataLoader()
    assert loader.dataset_name == "codecontests"


def test_parse_hf_tests_no_test_key():
    loader = DataLoader()
    result = loader._parse_hf_tests({"prompt": "def solve(): pass"})
    assert result == []


def test_parse_hf_tests_with_test_string():
    loader = DataLoader()
    example = {"prompt": "def solve(): pass", "test": "assert solve(2) == 4"}
    result = loader._parse_hf_tests(example)
    assert result == [{"input": "", "expected": "assert solve(2) == 4"}]
