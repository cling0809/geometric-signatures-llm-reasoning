#!/usr/bin/env python3
"""Write a completed-run, multi-EOS generation sidecar audit."""

from __future__ import annotations

import argparse

from geoprobe.revision.runtime_generation_audit import write_generation_runtime_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--generation-config", required=True)
    args = parser.parse_args()
    output = write_generation_runtime_audit(args.run, args.generation_config)
    print(output)


if __name__ == "__main__":
    main()
