"""Smoke tests for _salvage_complete_objects."""
import sys
import json
sys.path.insert(0, "src")
from utils.bedrock import BedrockClient


def test_no_bracket():
    assert BedrockClient._salvage_complete_objects("no json here") == "[]"


def test_empty_array():
    assert BedrockClient._salvage_complete_objects("[]") == "[]"


def test_complete_objects():
    result = BedrockClient._salvage_complete_objects('[{"a": 1}, {"b": 2}]')
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0] == {"a": 1}
    assert parsed[1] == {"b": 2}


def test_truncated_mid_object():
    result = BedrockClient._salvage_complete_objects('[{"a": 1}, {"b": 2}, {"c": ')
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0] == {"a": 1}
    assert parsed[1] == {"b": 2}


def test_truncated_before_any_complete():
    assert BedrockClient._salvage_complete_objects('[{"a":') == "[]"


def test_nested_braces_in_strings():
    result = BedrockClient._salvage_complete_objects('[{"a": "val with {braces}"}]')
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["a"] == "val with {braces}"


def test_escaped_quotes_in_strings():
    result = BedrockClient._salvage_complete_objects('[{"a": "val with \\"quotes\\""}]')
    parsed = json.loads(result)
    assert len(parsed) == 1


def test_returns_empty_when_no_complete_objects():
    """Requirement 2.3: return empty list when no complete objects found."""
    assert BedrockClient._salvage_complete_objects("") == "[]"
    assert BedrockClient._salvage_complete_objects("[") == "[]"
    assert BedrockClient._salvage_complete_objects("[{") == "[]"
    assert BedrockClient._salvage_complete_objects('[{"key": "val') == "[]"
