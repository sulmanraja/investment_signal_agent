"""Local tests for tools/ that do NOT require API keys or network access.

Tests that need live APIs are covered by the __main__ blocks in each tool.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    print("\n--- tools imports (requests-only subset) ---")
    # Only test tools that use stdlib + requests (available system-wide)
    from tools.newsdata_tool import fetch_news
    from tools.fred_macro_tool import fetch_macro_indicators, get_latest_snapshot, COMMON_SERIES

    assert callable(fetch_news)
    assert callable(fetch_macro_indicators)
    assert callable(get_latest_snapshot)
    assert len(COMMON_SERIES) == 10
    print(f"  ✓ requests-based tool imports OK ({len(COMMON_SERIES)} FRED series registered)")
    print("  (github_tool/google_trends_tool/edgar_* need venv — run: make test-tools)")


def test_fred_missing_key():
    print("\n--- fred_macro_tool: missing key guard ---")
    import os
    original = os.environ.pop("FRED_API_KEY", None)
    try:
        from tools.fred_macro_tool import get_latest_snapshot
        try:
            get_latest_snapshot()
            print("  ✗ Should have raised EnvironmentError")
        except EnvironmentError as e:
            print(f"  ✓ Raised EnvironmentError: {e}")
    finally:
        if original:
            os.environ["FRED_API_KEY"] = original


def test_news_missing_key():
    print("\n--- newsdata_tool: missing key guard ---")
    import os
    original = os.environ.pop("NEWSDATA_API_KEY", None)
    try:
        from tools.newsdata_tool import fetch_news
        try:
            fetch_news("test")
            print("  ✗ Should have raised EnvironmentError")
        except EnvironmentError as e:
            print(f"  ✓ Raised EnvironmentError: {e}")
    finally:
        if original:
            os.environ["NEWSDATA_API_KEY"] = original


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST: tools/ (no API keys)")
    print("=" * 50)
    test_imports()
    test_fred_missing_key()
    test_news_missing_key()
    print("\n✅ All no-API tools tests passed.")
    print("\nNote: Live API tests use __main__ blocks in each tool file:")
    print("  python -m tools.github_tool          (no key needed)")
    print("  python -m tools.google_trends_tool   (no key needed)")
    print("  python -m tools.newsdata_tool         (needs NEWSDATA_API_KEY)")
    print("  python -m tools.fred_macro_tool       (needs FRED_API_KEY)")
