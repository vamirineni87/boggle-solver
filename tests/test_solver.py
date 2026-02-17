import time
from app.solver import Trie, load_trie, solve


def _make_trie(words: list[str]) -> Trie:
    trie = Trie()
    for w in words:
        trie.insert(w.upper())
    return trie


def test_basic_solve():
    board = [
        ["C", "A", "T", "S"],
        ["R", "E", "P", "O"],
        ["B", "O", "N", "E"],
        ["D", "I", "G", "S"],
    ]
    words = ["CAT", "CATS", "CAR", "CARE", "BONE", "BONES", "REP", "PEN", "PONE",
             "DIG", "DIGS", "ONE", "ONES", "APE", "NOD", "NOG", "SON", "REPO",
             "OPEN", "NOPE", "PEON", "SING", "SIGN"]
    trie = _make_trie(words)
    result = solve(board, 4, trie)
    assert "CAT" in result
    assert "BONE" in result
    assert "CATS" in result
    # Words not reachable via adjacency should not appear
    # All returned words must be in the dictionary
    for w in result:
        assert w in [x.upper() for x in words]


def test_qu_cell():
    board = [
        ["QU", "I", "T"],
        ["E",  "S", "A"],
        ["N",  "D", "R"],
    ]
    words = ["QUIT", "QUITE", "QUEST", "QUITS", "SAT", "SET", "TEN", "DEN", "STAR"]
    trie = _make_trie(words)
    result = solve(board, 3, trie)
    assert "QUIT" in result


def test_no_revisit():
    """A word requiring revisiting a cell should not be found."""
    board = [
        ["A", "B"],
        ["C", "D"],
    ]
    # "ABA" requires revisiting cell (0,0) — should not be found
    trie = _make_trie(["ABA", "AB", "ABC"])
    result = solve(board, 2, trie)
    assert "ABA" not in result
    assert "AB" in result


def test_empty_results_for_no_matches():
    board = [["Z", "Z"], ["Z", "Z"]]
    trie = _make_trie(["CAT", "DOG"])
    result = solve(board, 2, trie)
    assert result == []


def test_max_results_cap():
    board = [
        ["C", "A", "T", "S"],
        ["R", "E", "P", "O"],
        ["B", "O", "N", "E"],
        ["D", "I", "G", "S"],
    ]
    # Use a big word list
    words = ["CAT", "CATS", "CAR", "CARE", "BONE", "BONES", "REP", "PEN", "PONE",
             "DIG", "DIGS", "ONE", "ONES", "APE", "NOD", "NOG", "SON", "REPO"]
    trie = _make_trie(words)
    result = solve(board, 4, trie, max_results=3)
    assert len(result) <= 3


def test_sort_order():
    board = [
        ["C", "A", "T", "S"],
        ["R", "E", "P", "O"],
        ["B", "O", "N", "E"],
        ["D", "I", "G", "S"],
    ]
    words = ["CAT", "CATS", "BONE", "BONES", "REP"]
    trie = _make_trie(words)
    result = solve(board, 4, trie)
    # Longest first
    for i in range(len(result) - 1):
        assert len(result[i]) >= len(result[i + 1]) or (
            len(result[i]) == len(result[i + 1]) and result[i] <= result[i + 1]
        )


def test_performance_with_full_dictionary(tmp_path):
    """Solve a 4x4 board with a decent-size dictionary under 500ms."""
    # Create a small but non-trivial dictionary
    dict_file = tmp_path / "dict.txt"
    words = []
    # Generate many 3-6 letter words from common letters
    import itertools
    letters = "ABCDEFGHIJKLMNOPRSTUE"
    for length in range(3, 5):
        for combo in itertools.combinations(letters, length):
            words.append("".join(combo))
            if len(words) > 5000:
                break
        if len(words) > 5000:
            break
    dict_file.write_text("\n".join(words))
    trie = load_trie(str(dict_file), min_length=3)

    board = [
        ["T", "A", "P", "E"],
        ["I", "N", "S", "O"],
        ["E", "D", "R", "L"],
        ["K", "G", "H", "M"],
    ]

    start = time.perf_counter()
    result = solve(board, 4, trie)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"Solver took {elapsed:.3f}s (expected <0.5s)"
    assert len(result) > 0
