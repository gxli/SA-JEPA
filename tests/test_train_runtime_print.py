from src.train import _format_active_loss_terms, _format_progress_line


def test_runtime_loss_print_keeps_weighted_active_terms():
    terms = _format_active_loss_terms(
        total=2.3,
        prediction=0.01,
        prediction_weight=50.0,
        spread=0.2,
        spread_weight=5.0,
        symmetry=0.03,
        symmetry_weight=0.003,
        vicreg_var=0.0,
        vicreg_var_weight=0.0,
        vicreg_cov=0.0,
        vicreg_cov_weight=0.0,
    )

    assert terms["pred"] == "0.0100"
    assert terms["wpred"] == "0.5000"
    assert terms["spread"] == "0.2000(active)"
    assert terms["wspread"] == "1.0000"
    assert terms["sym"] == "0.0300"
    assert terms["wsym"] == "9.000e-05"
    assert "vicvar" not in terms
    assert "viccov" not in terms

    line = _format_progress_line(
        "[batch]",
        terms,
        {"ctx_std": "0.9", "ctx_effrank": "1.2", "valid": "1.0"},
        {"lr": "1e-4"},
    )
    assert "wpred=0.5000" in line
    assert "wspread=1.0000" in line
    assert "wsym=9.000e-05" in line


def test_runtime_loss_print_hides_zero_weight_optional_terms():
    terms = _format_active_loss_terms(
        total=1.0,
        prediction=0.02,
        prediction_weight=50.0,
        spread=0.0,
        spread_weight=0.0,
        symmetry=0.0,
        symmetry_weight=0.0,
        vicreg_var=0.4,
        vicreg_var_weight=0.0,
        vicreg_cov=100.0,
        vicreg_cov_weight=0.0,
    )

    assert set(terms) == {"total", "pred", "wpred"}
