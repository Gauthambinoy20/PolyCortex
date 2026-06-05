import pytest

from polymarket_agent.infra.env_validator import validate_private_key

_VALID_KEY = "4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"


def test_valid_key_returns_cleaned_hex():
    out = validate_private_key(_VALID_KEY)
    assert out == _VALID_KEY
    assert len(out) == 64


def test_strips_0x_prefix():
    out = validate_private_key("0x" + _VALID_KEY)
    assert out == _VALID_KEY


def test_strips_whitespace():
    out = validate_private_key("  " + _VALID_KEY + "\n")
    assert out == _VALID_KEY


def test_rejects_wrong_length():
    with pytest.raises(ValueError, match="64 hex chars"):
        validate_private_key("abc")


def test_rejects_non_hex():
    bad = "z" * 64
    with pytest.raises(ValueError, match="non-hex"):
        validate_private_key(bad)
