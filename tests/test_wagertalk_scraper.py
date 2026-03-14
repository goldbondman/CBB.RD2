from datetime import date

from ingestion import wagertalk_scraper as ws


def test_scrape_range_skips_clear_offseason_window(tmp_path, monkeypatch):
    scraper = ws.WagerTalkScraper(output_path=tmp_path / "wagertalk.csv", delay=0.0)
    called_dates = []

    def fake_scrape_date(date_str):
        called_dates.append(date_str)
        return []

    monkeypatch.setattr(scraper, "scrape_date", fake_scrape_date)
    monkeypatch.setattr(scraper, "_write_rows", lambda rows, append=False: None)

    scraper.scrape_range(date(2025, 4, 14), date(2025, 11, 2))

    assert called_dates == [
        "2025-04-14",
        "2025-11-01",
        "2025-11-02",
    ]


def test_is_probable_offseason_day_uses_mid_april_boundary():
    assert ws.is_probable_offseason_day(date(2025, 4, 14)) is False
    assert ws.is_probable_offseason_day(date(2025, 4, 15)) is True
    assert ws.is_probable_offseason_day(date(2025, 10, 31)) is True
    assert ws.is_probable_offseason_day(date(2025, 11, 1)) is False


def test_scrape_date_reuses_single_warm_session(tmp_path, monkeypatch):
    class FakeSession:
        def __init__(self):
            self._warmed = False
            self.warm_calls = []
            self.requests = []

        def warm_up(self, date_str):
            self.warm_calls.append(date_str)
            self._warmed = True

        def get_data(self, date_str, data_file):
            self.requests.append((date_str, data_file))
            if not self._warmed:
                self.warm_up(date_str)
            return "schedule"

    scraper = ws.WagerTalkScraper(output_path=tmp_path / "wagertalk.csv", delay=0.0)
    scraper.session = FakeSession()

    monkeypatch.setattr(ws, "parse_schedule", lambda _text: [])

    scraper.scrape_date("2026-01-01")
    scraper.scrape_date("2026-01-02")

    assert scraper.session.warm_calls == ["2026-01-01"]
