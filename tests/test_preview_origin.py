import pathlib

BACKEND_FILE = pathlib.Path(__file__).resolve().parent.parent / "backend.py"


def test_preview_origin_regex_present():
    content = BACKEND_FILE.read_text()
    expected = 'r"^https://.*oksayless.*\\.onrender\\.com$"'
    assert expected in content
    assert "allow_origin_regex" in content
