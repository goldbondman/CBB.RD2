"""Edge-history lifecycle entrypoint."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage edge history rows.")
    parser.add_argument("--mode", choices=["append", "grade"], default="append")
    parser.parse_args()


if __name__ == "__main__":
    main()
