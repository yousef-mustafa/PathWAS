import pandas as pd
from pathwas.pas.pas import compute_pas


def test_compute_pas_mean():
    expr = pd.DataFrame({'g1': [1, 2], 'g2': [3, 4]}, index=['s1', 's2'])
    pathways = {'pw1': ['g1', 'g2']}
    pas, weights = compute_pas(expr, pathways, method='mean')
    assert weights is None
    assert pas.loc['s1', 'pw1'] == 2
    assert pas.loc['s2', 'pw1'] == 3


def test_compute_pas_activity_weighted():
    expr = pd.DataFrame({'g1': [1, 2, 3], 'g2': [2, 4, 6]}, index=['s1', 's2', 's3'])
    pathways = {'pw1': ['g1', 'g2']}
    pas, weights = compute_pas(expr, pathways, method='activity_weighted', corr_method='pearson')
    assert set(weights['pw1'].keys()) == {'g1', 'g2'}
    # correlation is 1 for identical profiles -> equal weights
    assert abs(weights['pw1']['g1'] - weights['pw1']['g2']) < 1e-6
    assert pas.loc['s1', 'pw1'] == 1.5
    assert pas.loc['s2', 'pw1'] == 3.0
    assert pas.loc['s3', 'pw1'] == 4.5

from pathwas.io.data_prep import convert_gene_ids


class DummyMG:
    def querymany(self, genes, scopes=None, fields=None, species=None):
        return [{"query": g, "ensembl": f"ENS{g}"} for g in genes]


def test_convert_gene_ids(monkeypatch):
    import types
    import pathwas.io.data_prep as dp

    monkeypatch.setattr(dp, 'mygene', types.SimpleNamespace(MyGeneInfo=lambda: DummyMG()))
    expr = pd.DataFrame({'TP53': [1, 2], '1234': [3, 4]}, index=['s1', 's2'])
    conv = convert_gene_ids(expr)
    assert set(conv.columns) == {'ENSTP53', 'ENS1234'}
