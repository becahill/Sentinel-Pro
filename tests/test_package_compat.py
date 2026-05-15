import importlib


def test_api_shim_exports_package_app(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB_PATH", str(tmp_path / "audit_logs.db"))
    monkeypatch.setenv("SENTINEL_DISABLE_TOXICITY", "1")

    legacy_api = importlib.import_module("api")
    legacy_api = importlib.reload(legacy_api)
    package_api = importlib.import_module("sentinel_pro.api.app")

    assert legacy_api.app is package_api.app


def test_root_module_shims_export_package_objects():
    from auditor import AuditEngine as LegacyAuditEngine
    from sentinel_pro.core.auditor import AuditEngine as PackageAuditEngine
    from sentinel_pro.core.signals import SignalDetector as PackageSignalDetector
    from signals import SignalDetector as LegacySignalDetector

    assert LegacyAuditEngine is PackageAuditEngine
    assert LegacySignalDetector is PackageSignalDetector
