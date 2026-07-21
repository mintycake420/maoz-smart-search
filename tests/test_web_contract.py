from pathlib import Path


def test_html_is_hebrew_rtl_and_has_no_external_assets() -> None:
    html = Path("maoz_search/templates/index.html").read_text(encoding="utf-8")
    assert '<html lang="he" dir="rtl"' in html
    assert "https://" not in html
    assert "http://" not in html


def test_frontend_does_not_inject_api_text_with_inner_html() -> None:
    script = Path("maoz_search/static/app.js").read_text(encoding="utf-8")
    assert ".innerHTML" not in script
