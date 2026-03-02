import sys
from pathlib import Path
import importlib

# Add project root to sys.path so "app" can be imported
sys.path.append(str(Path(__file__).parent.parent))

def test_import_app_main():
    # ensure app.main imports without raising
    mod = importlib.import_module("app.main")
    assert mod is not None

def test_app_exposes_entrypoint():
    mod = importlib.import_module("app.main")
    assert any(hasattr(mod, name) for name in ("app", "create_app", "main")), \
        "app.main should expose one of: app, create_app, main"