from datetime import date
import importlib


def test_cli_forwards_parallel_enrichment_workers(monkeypatch):
    module = importlib.import_module("src.pipeline.run_task1")
    captured = {}

    monkeypatch.setattr(module, "load_config", lambda _path: {"validation": {}})

    def fake_run_task1(**kwargs):
        captured.update(kwargs)
        return {"status": "passed"}

    monkeypatch.setattr(module, "run_task1", fake_run_task1)
    result = module.main([
        "--source", "cnbc",
        "--start-date", "2021-09-01",
        "--end-date", "2026-09-01",
        "--full-range",
        "--skip-jisdor-clean",
        "--enrichment-workers", "4",
    ])

    assert result == 0
    assert captured["start_date"] == date(2021, 9, 1)
    assert captured["end_date"] == date(2026, 9, 1)
    assert captured["enrichment_workers"] == 4


def test_cli_enrichment_workers_default_is_one(monkeypatch):
    module = importlib.import_module("src.pipeline.run_task1")
    captured = {}
    monkeypatch.setattr(module, "load_config", lambda _path: {"validation": {}})
    monkeypatch.setattr(
        module, "run_task1",
        lambda **kwargs: captured.update(kwargs) or {"status": "passed"},
    )

    assert module.main([
        "--start-date", "2021-09-01", "--end-date", "2021-09-07",
    ]) == 0
    assert captured["enrichment_workers"] == 1
