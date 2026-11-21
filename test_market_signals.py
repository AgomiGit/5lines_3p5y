import unittest
from unittest.mock import patch

from market_signals import (
    fear_greed_band,
    forward_pe_band,
    get_forward_pe_data,
    parse_forward_pe_html,
    vix_band,
)


class MarketSignalTests(unittest.TestCase):
    def test_signal_band_boundaries(self):
        self.assertEqual(fear_greed_band(24.9)[0], "極度恐慌（0～<25）")
        self.assertEqual(fear_greed_band(25)[0], "恐慌（25～<45）")
        self.assertEqual(vix_band(35.89)[0], "市場恐慌（30～<40）")
        self.assertEqual(vix_band(40)[0], "極度恐慌（≥40）")
        self.assertEqual(forward_pe_band(16.42)[0], "非常便宜（低於 18 倍）")
        self.assertEqual(forward_pe_band(18)[0], "相對便宜（18～<22 倍）")
        self.assertEqual(fear_greed_band(None), ("資料暫缺", "tone-muted"))

    def test_parse_forward_pe_card(self):
        sample = """
        <section>
          <h2>Nasdaq-100 Forward P/E</h2>
          <p>16.42<span>×</span></p>
          <p>6.09% Earnings yield</p>
          <div>As of <span>Nov 20, 2026</span><span>−1.44</span></div>
        </section>
        """
        self.assertEqual(
            parse_forward_pe_html(sample),
            {
                "value": 16.42,
                "change": -1.44,
                "previous": 17.86,
                "date": "2026-11-20",
            },
        )

    @patch("market_signals.urlopen", side_effect=OSError("offline"))
    def test_forward_pe_failure_returns_empty_signal(self, _mock_urlopen):
        self.assertEqual(
            get_forward_pe_data(),
            {"value": None, "change": None, "previous": None, "date": None},
        )


if __name__ == "__main__":
    unittest.main()
