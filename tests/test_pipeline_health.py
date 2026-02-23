from pathlib import Path


def test_pipeline_packages_exist() -> None:
    required = [
        Path('config/__init__.py'),
        Path('ingestion/__init__.py'),
        Path('features/__init__.py'),
        Path('models/__init__.py'),
        Path('evaluation/__init__.py'),
        Path('enrichment/__init__.py'),
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Missing package markers: {missing}"
