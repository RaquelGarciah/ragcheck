"""Tests de features: caso feliz, caso límite y determinismo por función."""

import pandas as pd

from ragcheck.features import (
    answer_len,
    containment,
    extract_features,
    jaccard,
    num_overlap,
    tfidf_cos,
)


def test_containment_caso_feliz():
    # Tres de las cuatro palabras de la respuesta están en la fuente.
    r = "the cat sat here"
    s = "the cat sat on the mat"
    assert containment(r, s) == 3 / 4


def test_containment_respuesta_vacia():
    assert containment("", "algo de fuente") == 0.0


def test_jaccard_simetria_y_rango():
    r, s = "a b c", "b c d"
    # |R ∩ F| = 2 ({b,c}), |R ∪ F| = 4 ({a,b,c,d}).
    assert jaccard(r, s) == 2 / 4
    assert jaccard(r, s) == jaccard(s, r)


def test_jaccard_ambos_vacios():
    assert jaccard("", "") == 0.0


def test_num_overlap_caza_cifras_inventadas():
    # 2020 está en la fuente; 1999 es inventado -> 1 de 2.
    assert num_overlap("en 2020 y 1999", "el año 2020") == 1 / 2


def test_num_overlap_sin_numeros():
    # Sin números en la respuesta, la señal numérica no se viola.
    assert num_overlap("sin cifras aqui", "tampoco alli") == 1.0


def test_tfidf_cos_identico_y_disjunto():
    assert tfidf_cos("mismo texto", "mismo texto") > 0.99
    assert tfidf_cos("perro gato", "avion barco") == 0.0


def test_answer_len_cuenta_tokens():
    assert answer_len("una respuesta de cinco tokens exactos", "") == 6.0


def test_features_deterministas():
    r, s = "the cat sat here", "the cat sat on the mat"
    for f in (containment, jaccard, tfidf_cos, num_overlap, answer_len):
        assert f(r, s) == f(r, s)


def test_extract_features_esquema():
    df = pd.DataFrame(
        {"response": ["the cat", "1999 aqui"], "source": ["the cat mat", "año 2020"]}
    )
    feats = extract_features(df)
    assert list(feats.columns) == [
        "containment",
        "jaccard",
        "tfidf_cos",
        "num_overlap",
        "answer_len",
    ]
    assert len(feats) == 2
    assert feats.index.equals(df.index)
