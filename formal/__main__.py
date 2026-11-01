"""Build the installed proof library explicitly; verification never builds implicitly."""

import subprocess
from pathlib import Path


def main() -> None:
    subprocess.run(["make", "-C", str(Path(__file__).resolve().parent), "build"], check=True)


if __name__ == "__main__":
    main()
