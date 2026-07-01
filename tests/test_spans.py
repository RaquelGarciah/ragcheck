"""Tests del parseo de spans alucinados (offsets correctos, span vacío)."""

from ragcheck.spans import parse_spans


def test_parse_spans_offsets():
    raw = '[{"start": 3, "end": 8, "text": "hola", "label_type": "Evident Conflict"}]'
    spans = parse_spans(raw)
    assert len(spans) == 1
    assert (spans[0]["start"], spans[0]["end"]) == (3, 8)
    assert spans[0]["label_type"] == "Evident Conflict"


def test_parse_spans_respuesta_limpia():
    assert parse_spans("[]") == []
    assert parse_spans("") == []


def test_parse_spans_acepta_lista():
    spans = parse_spans([{"start": 0, "end": 1}])
    assert spans == [{"start": 0, "end": 1}]


def test_parse_spans_no_evaluable():
    # Cadena no-JSON: se degrada a lista vacía, no revienta.
    assert parse_spans("no soy json") == []
