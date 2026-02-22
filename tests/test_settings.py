from app.settings import Settings, EDITABLE_FIELDS, update_settings, get_editable_settings


def _fresh_settings() -> Settings:
    """Create a fresh Settings instance for testing."""
    return Settings()


def test_editable_fields_exist_on_settings():
    """All editable fields must be actual attributes on Settings."""
    cfg = _fresh_settings()
    for field_name in EDITABLE_FIELDS:
        assert hasattr(cfg, field_name), f"{field_name} not found on Settings"


def test_get_editable_settings():
    cfg = _fresh_settings()
    result = get_editable_settings(cfg)
    assert set(result.keys()) == set(EDITABLE_FIELDS.keys())
    assert result["COMMON_WORDS_ONLY"] == cfg.COMMON_WORDS_ONLY
    assert result["MAX_RESULTS"] == cfg.MAX_RESULTS


def test_update_int_field():
    cfg = _fresh_settings()
    errors = update_settings(cfg, NOTIFY_WORDS_PER_GROUP=7)
    assert errors == {}
    assert cfg.NOTIFY_WORDS_PER_GROUP == 7


def test_update_bool_field():
    cfg = _fresh_settings()
    original = cfg.DEBUG
    errors = update_settings(cfg, DEBUG=not original)
    assert errors == {}
    assert cfg.DEBUG is (not original)


def test_update_bool_from_json_true():
    """JSON sends true/false as Python bool, not string."""
    cfg = _fresh_settings()
    errors = update_settings(cfg, DEBUG=True)
    assert errors == {}
    assert cfg.DEBUG is True


def test_update_bool_from_string():
    cfg = _fresh_settings()
    errors = update_settings(cfg, DEBUG="true")
    assert errors == {}
    assert cfg.DEBUG is True

    errors = update_settings(cfg, DEBUG="false")
    assert errors == {}
    assert cfg.DEBUG is False


def test_update_float_field():
    cfg = _fresh_settings()
    errors = update_settings(cfg, OCR_CONFIDENCE_THRESHOLD=0.5)
    assert errors == {}
    assert cfg.OCR_CONFIDENCE_THRESHOLD == 0.5


def test_update_string_field():
    cfg = _fresh_settings()
    errors = update_settings(cfg, NTFY_TOPIC="test-topic")
    assert errors == {}
    assert cfg.NTFY_TOPIC == "test-topic"


def test_update_multiple_fields():
    cfg = _fresh_settings()
    errors = update_settings(cfg, MAX_RESULTS=10, MIN_WORD_LENGTH=4, NTFY_TOPIC="multi")
    assert errors == {}
    assert cfg.MAX_RESULTS == 10
    assert cfg.MIN_WORD_LENGTH == 4
    assert cfg.NTFY_TOPIC == "multi"


def test_update_non_editable_field_returns_error():
    cfg = _fresh_settings()
    errors = update_settings(cfg, WARP_SIZE=800)
    assert "WARP_SIZE" in errors


def test_update_unknown_field_returns_error():
    cfg = _fresh_settings()
    errors = update_settings(cfg, NONEXISTENT_FIELD=42)
    assert "NONEXISTENT_FIELD" in errors


def test_update_partial_error():
    """Valid fields update even when invalid fields are present."""
    cfg = _fresh_settings()
    errors = update_settings(cfg, MAX_RESULTS=25, BAD_FIELD="nope")
    assert "BAD_FIELD" in errors
    assert cfg.MAX_RESULTS == 25


def test_common_words_default_true():
    cfg = _fresh_settings()
    assert cfg.COMMON_WORDS_ONLY is True


def test_trie_reload_on_common_words_change():
    """Changing COMMON_WORDS_ONLY should produce different trie sizes."""
    from app.solver import load_trie

    cfg = _fresh_settings()

    # Load with common words (default)
    cfg.COMMON_WORDS_ONLY = True
    dict_path = cfg.DICTIONARY_COMMON_PATH if cfg.COMMON_WORDS_ONLY else cfg.DICTIONARY_PATH
    trie_common = load_trie(str(dict_path), cfg.MIN_WORD_LENGTH)

    # Load with full dictionary
    cfg.COMMON_WORDS_ONLY = False
    dict_path = cfg.DICTIONARY_COMMON_PATH if cfg.COMMON_WORDS_ONLY else cfg.DICTIONARY_PATH
    trie_full = load_trie(str(dict_path), cfg.MIN_WORD_LENGTH)

    # Full dictionary should find more words than common
    board = [
        ["C", "A", "T", "S"],
        ["R", "E", "P", "O"],
        ["B", "O", "N", "E"],
        ["D", "I", "G", "S"],
    ]
    from app.solver import solve
    words_common, _ = solve(board, 4, trie_common, max_results=0)
    words_full, _ = solve(board, 4, trie_full, max_results=0)

    assert len(words_full) > len(words_common)


def test_trie_reload_on_min_word_length_change():
    """Changing MIN_WORD_LENGTH should affect which words appear."""
    from app.solver import load_trie, solve

    cfg = _fresh_settings()
    board = [
        ["C", "A", "T", "S"],
        ["R", "E", "P", "O"],
        ["B", "O", "N", "E"],
        ["D", "I", "G", "S"],
    ]

    trie_3 = load_trie(str(cfg.DICTIONARY_PATH), min_length=3)
    words_3, _ = solve(board, 4, trie_3, max_results=0)

    trie_5 = load_trie(str(cfg.DICTIONARY_PATH), min_length=5)
    words_5, _ = solve(board, 4, trie_5, max_results=0)

    # With min_length=3, there should be short words
    has_short = any(len(w) < 5 for w in words_3)
    assert has_short, "min_length=3 should include words shorter than 5"

    # With min_length=5, no short words
    all_long = all(len(w) >= 5 for w in words_5)
    assert all_long, "min_length=5 should exclude words shorter than 5"

    assert len(words_3) > len(words_5)
