from spotXAI import example_analyzer

import pytest

import torch


def test_IG():
    """
    Test the IntegratedGradients attribution method.
    """
    analyzer = example_analyzer()
    n_rel = 5
    df_ig = analyzer.get_n_most_sig_features(
        n_rel=n_rel, attr_method="IntegratedGradients", baseline=None, abs_attr=True
    )
    assert len(df_ig) == n_rel


# def test_DL():
#        """"
#        Test the DeepLift attribution method
#        Test the length of the returned data frame.
#        """
#        analyzer = example_analyzer()
#        n_rel = 5
#        df_dl = analyzer.get_n_most_sig_features(n_rel=n_rel, attr_method="DeepLift", baseline=None, abs_attr=True)
#        assert(len(df_dl) == n_rel)


def test_GS():
    """ "
    Test the GradientShap attribution method.
    Test the length of the returned data frame.
    Test if the method raises an error if no baseline is defined.
    """
    analyzer = example_analyzer()
    n_rel = 5
    baseline = torch.Tensor([[0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2]])
    df_gs = analyzer.get_n_most_sig_features(n_rel=n_rel, attr_method="GradientShap", baseline=baseline, abs_attr=True)
    assert len(df_gs) == n_rel
    with pytest.raises(ValueError):
        df_gs = analyzer.get_n_most_sig_features(n_rel=n_rel, attr_method="GradientShap", baseline=None, abs_attr=True)


def test_FA():
    """ "
    Test the DeepLift attribution method
    Test the length of the returned data frame.
    """
    analyzer = example_analyzer()
    n_rel = 5
    df_dl = analyzer.get_n_most_sig_features(n_rel=n_rel, attr_method="FeatureAblation", baseline=None, abs_attr=True)
    assert len(df_dl) == n_rel


def test_undefined_attribution():
    """ "
    Test the error response if an unsupported attribution method is selected.
    """
    analyzer = example_analyzer()
    n_rel = 5
    with pytest.raises(ValueError):
        df_dl = analyzer.get_n_most_sig_features(
            n_rel=n_rel, attr_method="unknownAttributionMethod", baseline=None, abs_attr=True
        )
