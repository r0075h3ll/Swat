from swat.levenshtein import distance


def test_identical_strings():
    assert distance("paypal.com", "paypal.com") == 0


def test_single_substitution():
    assert distance("paypal.com", "paypa1.com") == 1


def test_empty_strings():
    assert distance("", "abc") == 3
    assert distance("abc", "") == 3
    assert distance("", "") == 0


def test_classic_example():
    assert distance("kitten", "sitting") == 3


def test_transposition_adjacent_chars():
    # not treated specially (this is edit distance, not Damerau-Levenshtein)
    assert distance("ab", "ba") == 2
