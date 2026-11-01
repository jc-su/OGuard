"""Packaged raw regression observations, independent of local experiment archives."""

import gzip
import json
from pathlib import Path

import pytest


@pytest.fixture
def raw_record():
    def read(name):
        return json.loads(
            gzip.decompress((Path(__file__).with_name("fixtures") / (name + ".gz")).read_bytes())
        )

    return read
