import pytest

def pytest_sessionfinish(session, exitstatus):
    """Exit with 0 if no tests were collected.
    
    This hook is called after the entire test run is finished, and is used to
    override the exit status of the test run. If no tests were collected, the
    exit status is set to 0 (indicating success) instead of 5 (indicating no
    tests were collected). Otherwise, the template repository test actionwould 
    be marked as failing even though no tests were run.
    """
    if exitstatus == 5:  # no tests collected
        session.exitstatus = 0
