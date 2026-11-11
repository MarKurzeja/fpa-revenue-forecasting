import pandas as pd

from fpa.commentary import fmt_pct, fmt_usd, month_commentary


def test_formatting():
    assert fmt_usd(41_234_567) == "USD 41.2M"
    assert fmt_usd(-1_600_000, signed=True) == "-USD 1.6M"
    assert fmt_usd(350_000, signed=True) == "+USD 350K"
    assert fmt_usd(1_500) == "USD 1.5K"
    assert fmt_usd(2.5e9) == "USD 2.50bn"
    assert fmt_usd(12) == "USD 12"
    assert fmt_pct(-0.031) == "3.1%" and fmt_pct(0.012, signed=True) == "+1.2%"


def test_month_commentary_example():
    seg = pd.DataFrame(
        {
            "state_id": ["CA", "TX"],
            "cat_id": ["FOODS", "FOODS"],
            "volume": [-1.2e6, -0.4e6],
            "mix": [0.0, 0.0],
            "price": [0.3e6, 0.1e6],
        }
    )
    text = month_commentary(
        2016,
        7,
        41.2e6,
        42.52e6,
        {"volume": -1.6e6, "mix": 0.0, "price": 0.4e6},
        seg,
        -0.012,
        -0.004,
    )
    assert text.startswith("Fiscal month 7 (FY2016) revenue was USD 41.2M, 3.1% below budget")
    assert "Volume explains -USD 1.6M, mostly FOODS in CA." in text
    assert "Price added USD 400K" in text
    assert "1.2% below budget, down from 0.4% below last month" in text


def test_month_commentary_without_outlook():
    text = month_commentary(2016, 1, 100.0, 100.0, {}, pd.DataFrame(), None, None)
    assert "0.0% above budget" in text and "outlook" not in text
