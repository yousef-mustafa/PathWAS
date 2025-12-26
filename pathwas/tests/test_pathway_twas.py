import numpy as np
import pandas as pd

from pathwas.association.pathway_test import test_pathway_twas


def test_basic_twas():
    weights = pd.Series([0.5, 0.5], index=['snp1', 'snp2'])
    z = pd.Series([1.0, 2.0], index=['snp1', 'snp2'])
    ld = np.array([[1.0, 0.1], [0.1, 1.0]])

    z_pw, var_pw, contrib = test_pathway_twas(weights, z, ld, return_contributions=True)
    assert np.isclose(var_pw, 0.55)
    assert np.isclose(z_pw, 0.825)
    assert np.allclose(contrib.values, [0.275, 0.55])


def test_zero_variance():
    weights = pd.Series([0.0, 0.0], index=['a', 'b'])
    z = pd.Series([1.0, 2.0], index=['a', 'b'])
    ld = np.eye(2)

    z_pw, var_pw = test_pathway_twas(weights, z, ld)
    assert np.isnan(z_pw)
    assert var_pw == 0.0