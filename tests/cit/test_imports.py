import pytest


def test_public_exports():
    import rp_ci.cit as cit
    assert set(cit.__all__) <= set(dir(cit))
    for name in cit.__all__:
        assert getattr(cit, name) is not None
    with pytest.raises(AttributeError):
        getattr(cit, "not_an_export")
