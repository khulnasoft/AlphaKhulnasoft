from alphakhulnasoft.sandbox import Sandbox


def test_sandbox_simple_success():
    sb = Sandbox(timeout_seconds=2)
    code = "name = input(); print(f'Hello {name}')"
    test_cases = [{"input": "World", "expected": "Hello World"}]

    pass_rate, log = sb.run_tests(code, test_cases)
    assert pass_rate == 1.0
    assert log == ""


def test_sandbox_failure():
    sb = Sandbox(timeout_seconds=2)
    code = "print('Wrong Output')"
    test_cases = [{"input": "input", "expected": "Expected Output"}]

    pass_rate, log = sb.run_tests(code, test_cases)
    assert pass_rate == 0.0
    assert "Failed" in log


def test_sandbox_timeout():
    sb = Sandbox(timeout_seconds=1)
    code = "import time; time.sleep(2)"
    test_cases = [{"input": "", "expected": ""}]

    pass_rate, log = sb.run_tests(code, test_cases)
    assert pass_rate == 0.0
    assert "Time Limit Exceeded" in log
