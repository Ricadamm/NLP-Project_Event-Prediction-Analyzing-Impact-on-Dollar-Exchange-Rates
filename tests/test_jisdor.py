from datetime import datetime
import hashlib
import json

import pandas as pd
import pytest

from src.preprocessing.clean_jisdor import (
    JisdorValidationError, clean_workbook, extract_table, locate_workbook,
    parse_date, parse_rate, validate_table,
)


def table(rows):
    return pd.DataFrame(rows, columns=["source_row", "date_raw", "jisdor_raw"])


def test_extract_named_columns_blank_rows_and_partial_rows():
    source = pd.DataFrame([
        ["BI export", None, None, None],
        [None, None, None, None],
        [" Kurs ", "NO", None, " TANGGAL "],
        [14284, 1, None, "9/1/2021 12:00:00 AM"],
        [None, None, None, None],
        [None, 2, None, "9/2/2021"],
    ])
    raw, metadata = extract_table({"Info": pd.DataFrame([["cover"]]), "Rates": source})
    assert raw["source_row"].tolist() == [4, 6]
    assert metadata["header_row"] == 3
    assert metadata["blank_rows_ignored"] == 1
    assert metadata["empty_columns"] == [3]
    with pytest.raises(JisdorValidationError) as failure:
        validate_table(raw)
    assert failure.value.report["missing_jisdor"] == 1


@pytest.mark.parametrize("value,expected", [
    ("9/1/2021 12:00:00 AM", "2021-09-01"),
    (datetime(2026, 9, 1, 13, 15), "2026-09-01"),
    ("2021-09-02", "2021-09-02"),
])
def test_date_parsing(value, expected):
    assert parse_date(value) == pd.Timestamp(expected)


@pytest.mark.parametrize("value", [44440, 44440.5, "44440", "20210901", "no-date", "2021-02-30", "2021-09-01T00:00:00Z", "2021-09", "9/1", "13/01/2021", "September"])
def test_invalid_dates(value):
    with pytest.raises((ValueError, TypeError)):
        parse_date(value)


@pytest.mark.parametrize("value,expected", [
    (14284, 14284), (14284.5, 14284.5), ("14.284", 14284),
    ("14,284", 14284), ("Rp 14.284,50", 14284.5),
    ("14,284.50", 14284.5), ("14\u00a0284", 14284),
    ("14284,50", 14284.5), ("'17727", 17727),
])
def test_numeric_artifacts(value, expected):
    assert parse_rate(value) == expected


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), "NaN", "inf", "garbage", True, 0, -1, "1,234.567", "1 23"])
def test_invalid_rates(value):
    with pytest.raises((ValueError, TypeError)):
        parse_rate(value)


def test_sort_preserves_observations_and_expected_values_only_compare():
    clean, report = validate_table(table([
        [6, "2021-09-06", 99999], [5, "2021-09-03", 123],
    ]))
    assert list(clean.columns) == ["date", "jisdor"]
    assert clean["date"].dt.strftime("%Y-%m-%d").tolist() == ["2021-09-03", "2021-09-06"]
    assert clean["jisdor"].tolist() == [123, 99999]
    assert report["row_count"] == 2
    assert report["chronologically_sorted"]
    assert "row_count" in report["expected_discrepancies"]


@pytest.mark.parametrize("rates", [[14000, 14000], [14000, 15000]])
def test_duplicate_dates_fail_without_removal(rates):
    with pytest.raises(JisdorValidationError) as failure:
        validate_table(table([[6, "2021-09-01", rates[0]], [7, "2021-09-01 12:00", rates[1]]]))
    assert failure.value.report["duplicate_dates"] == 1
    assert len(failure.value.report["duplicate_records"]) == 2


def test_missing_and_invalid_values_report_source_records():
    with pytest.raises(JisdorValidationError) as failure:
        validate_table(table([[6, None, 14000], [7, "2021-09-02", None], [8, 44440, "inf"]]))
    report = failure.value.report
    assert report["missing_dates"] == report["missing_jisdor"] == 1
    assert report["invalid_dates"] == report["invalid_jisdor"] == 1
    assert [record["source_row"] for record in report["invalid_records"]] == [6, 7, 8]


def test_ambiguous_headers_and_workbooks_require_selection(tmp_path):
    frame = pd.DataFrame([["Tanggal", "Kurs"], ["2021-09-01", 14284]])
    with pytest.raises(JisdorValidationError):
        extract_table({"one": frame, "two": frame})
    with pytest.raises(JisdorValidationError):
        extract_table({"one": pd.DataFrame([["Tanggal", "Tanggal", "Kurs"]])})
    (tmp_path / "one.xlsx").touch()
    assert locate_workbook(tmp_path).name == "one.xlsx"
    (tmp_path / "two.xlsx").touch()
    with pytest.raises(JisdorValidationError):
        locate_workbook(tmp_path)


def test_workbook_hash_output_and_failed_rerun(tmp_path):
    source = tmp_path / "source.xlsx"
    output = tmp_path / "clean.csv"
    qa = tmp_path / "qa.json"
    pd.DataFrame({"Tanggal": ["2021-09-02", "2021-09-01"], "Kurs": [14281, 14284]}).to_excel(source, index=False)
    original = source.read_bytes()
    _, report = clean_workbook(source, output, qa)
    assert source.read_bytes() == original
    assert report["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert output.read_text() == "date,jisdor\n2021-09-01,14284\n2021-09-02,14281\n"
    assert json.loads(qa.read_text())["status"] == "passed"
    pd.DataFrame({"Tanggal": ["2021-09-02"], "Kurs": [None]}).to_excel(source, index=False)
    with pytest.raises(JisdorValidationError):
        clean_workbook(source, output, qa)
    assert not output.exists()
    failed = json.loads(qa.read_text())
    assert failed["status"] == "failed"
    assert not failed["output_valid"]
    assert failed["invalid_records"][0]["source_row"] == 2


def test_outputs_cannot_overwrite_source(tmp_path):
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"unchanged")
    with pytest.raises(ValueError):
        clean_workbook(source, source, tmp_path / "qa.json")
    assert source.read_bytes() == b"unchanged"
