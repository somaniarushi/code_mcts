test = """
METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
"""
# For a string that looks like this, edit it to instead return the number of passed and failed tests
# The result string should look like:
"""
METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate) -> list[bool]:
    return [
        candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True,
        candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False,
        candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True,
        candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False,
        candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True,
        candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True,
        candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False
    ]
"""


print(edit_test_string(test))