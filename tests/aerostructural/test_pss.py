def test_aerostructural_import_does_not_import_pss():
    import awetrim.aerostructural as aerostructural

    assert hasattr(aerostructural, "PssKineticDampingSolver")
