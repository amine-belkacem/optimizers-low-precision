import matplotlib

matplotlib.use("Agg")  # no display in CI

import matplotlib.pyplot as plt
import pytest

from lpopt import plotting


def test_series_order_keeps_blue_and_orange_last():
    expected = ("#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#008300", "#2a78d6", "#eb6834")
    got = plotting.SERIES_COLORS
    assert got == expected, f"ECHEC: obtenu {got}, attendu {expected}"


def test_two_series_are_teal_then_violet():
    got = (plotting.color_for(0), plotting.color_for(1))
    expected = ("#1baf7a", "#6250d6")
    assert got == expected, f"ECHEC: obtenu {got}, attendu {expected}"


@pytest.mark.parametrize("index", [-1, 7])
def test_color_for_rejects_out_of_range(index):
    with pytest.raises(ValueError):
        plotting.color_for(index)


def test_each_optimizer_has_its_own_non_blue_non_orange_color():
    colors = list(plotting.OPTIMIZER_COLORS.values())
    assert len(set(colors)) == len(colors), f"ECHEC: obtenu {colors}, attendu 4 couleurs distinctes"
    forbidden = {plotting.BLUE, plotting.ORANGE}
    got = forbidden.intersection(colors)
    assert not got, f"ECHEC: obtenu {got}, attendu aucune couleur bleue ou orange"


def test_apply_style_sets_the_color_cycle():
    plotting.apply_style()
    got = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][0]
    expected = "#1baf7a"
    assert got == expected, f"ECHEC: obtenu {got}, attendu {expected}"


def test_save_figure_writes_a_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = plotting.save_figure(fig, tmp_path / "sub" / "fig.png")
    header = path.read_bytes()[:8]
    expected = b"\x89PNG\r\n\x1a\n"
    assert header == expected, f"ECHEC: obtenu {header!r}, attendu {expected!r}"
