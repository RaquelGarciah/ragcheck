"""Tests de features: caso feliz, caso límite y determinismo por función."""

import pandas as pd

from ragcheck.features import (
    FEATURES,
    SENTENCE_FEATURES,
    _sentence_features,
    answer_len,
    containment,
    extract_features,
    jaccard,
    neg_diff,
    novel_bigram,
    novel_trigram,
    num_context,
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


def test_novel_bigram_caza_recombinacion():
    # Todas las palabras están en la fuente, pero "26.2 hours" no es un bigrama
    # suyo (la fuente dice "26.2 miles"): el unigrama no lo ve, el bigrama sí.
    r = "26.2 hours"
    s = "the marathon is 26.2 miles and lasts hours"
    assert containment(r, s) == 1.0  # unigram ciego
    assert novel_bigram(r, s) == 1.0  # bigrama lo caza


def test_novel_bigram_texto_extractivo():
    # Respuesta copiada literalmente: ningún bigrama nuevo.
    assert novel_bigram("the cat sat", "the cat sat on the mat") == 0.0


def test_novel_bigram_respuesta_corta():
    assert novel_bigram("cat", "the cat") == 0.0  # menos de dos tokens


def test_novel_trigram_mas_estricto_que_bigram():
    r, s = "the cat sat quietly", "the cat ran and sat quietly"
    # "cat sat" es nuevo (bigram>0); a nivel trigram también hay novedad.
    assert novel_trigram(r, s) >= novel_bigram(r, s) or novel_trigram(r, s) > 0


def test_num_context_detecta_cambio_de_unidad():
    # 26.2 está en la fuente pero con "miles", no con "hours".
    assert num_context("ran 26.2 hours", "ran 26.2 miles") == 0.0
    # Mismo contexto -> anclado.
    assert num_context("ran 26.2 miles", "ran 26.2 miles today") == 1.0


def test_num_context_sin_numeros():
    assert num_context("sin cifras", "tampoco aqui") == 1.0


def test_neg_diff_polaridad():
    # La respuesta niega y la fuente no: diferencia positiva.
    assert neg_diff("he did not attend", "he attended the event") > 0
    # Sin negaciones en ninguno: cero.
    assert neg_diff("he attended", "he attended too") == 0.0


def test_features_deterministas():
    r, s = "the cat sat here", "the cat sat on the mat"
    for f in (containment, jaccard, tfidf_cos, num_overlap, answer_len,
              novel_bigram, novel_trigram, num_context, neg_diff):
        assert f(r, s) == f(r, s)


def test_sent_features_caza_frase_sin_respaldo():
    # Primera frase respaldada, segunda inventada de cero.
    r = "The cat sat on the mat. Dragons rule the distant galaxy."
    s = "The cat sat on the mat quietly."
    f = _sentence_features(r, s)
    # La frase peor sostenida arrastra el mínimo muy por debajo de la media.
    assert f["sent_cont_min"] < f["sent_cont_mean"]
    assert f["sent_frac_low"] == 0.5  # una de dos frases mal respaldada
    assert f["sent_sim_min"] < f["sent_sim_mean"]


def test_sent_features_respuesta_limpia_da_soporte_alto():
    r = "The cat sat on the mat."
    s = "The cat sat on the mat quietly in the sun."
    f = _sentence_features(r, s)
    assert f["sent_cont_min"] == 1.0  # todas las palabras están en la fuente
    assert f["sent_cont_std"] == 0.0  # una sola frase, sin dispersión


def test_sent_features_respuesta_vacia_neutra():
    f = _sentence_features("", "algo de fuente")
    assert f["sent_cont_min"] == 1.0 and f["sent_sim_min"] == 1.0
    assert f["sent_frac_low"] == 0.0


def test_sent_features_deterministas():
    r, s = "One grounded line. A wholly invented line here.", "One grounded line only."
    assert _sentence_features(r, s) == _sentence_features(r, s)


def test_extract_features_esquema():
    df = pd.DataFrame(
        {
            "output": ["the cat", "1999 aqui"],
            "context": ["the cat mat", "año 2020"],
            "task_type": ["QA", "Data2txt"],
        }
    )
    feats = extract_features(df)
    assert list(feats.columns) == [
        *FEATURES,
        *SENTENCE_FEATURES,
        "task_QA",
        "task_Summary",
        "task_Data2txt",
    ]
    assert len(feats) == 2
    assert feats.index.equals(df.index)
    # One-hot correcto: la primera fila es QA, la segunda Data2txt.
    assert feats["task_QA"].tolist() == [1.0, 0.0]
    assert feats["task_Data2txt"].tolist() == [0.0, 1.0]


def test_extract_features_sin_task_type():
    # Sin la columna task_type no se añaden las one-hot (contrato de inferencia).
    df = pd.DataFrame({"output": ["the cat"], "context": ["the cat mat"]})
    feats = extract_features(df)
    assert list(feats.columns) == [*FEATURES, *SENTENCE_FEATURES]
