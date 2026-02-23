"""Bias detection entrypoint."""

import argparse
import runpy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline bias detection checks.")
    parser.parse_args()
    runpy.run_module('evaluation.audit_feature_leakage', run_name='__main__')


if __name__ == "__main__":
    main()
