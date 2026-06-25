# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for vended storage-credential refresh (provider, PyArrow rebuild, REST wiring)."""

import pickle
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from pyiceberg.catalog.rest import RestCatalog
from pyiceberg.catalog.rest.scan_planning import StorageCredential
from pyiceberg.io import VendedCredentialsProvider
from pyiceberg.io.pyarrow import PyArrowFileIO


def test_provider_refreshes_when_near_expiry() -> None:
    calls = {"n": 0}

    def refresh() -> tuple[dict, int | None]:
        calls["n"] += 1
        return {"s3.access-key-id": "NEW"}, None

    provider = VendedCredentialsProvider({"s3.access-key-id": "OLD"}, expires_at_ms=0, refresh=refresh)
    assert provider.credentials()["s3.access-key-id"] == "NEW"
    assert provider.generation == 1
    assert calls["n"] == 1


def test_provider_skips_when_not_near_expiry() -> None:
    refresh = MagicMock()
    future = int(time.time() * 1000) + 3_600_000
    provider = VendedCredentialsProvider({"s3.access-key-id": "OLD"}, expires_at_ms=future, refresh=refresh)
    assert provider.credentials()["s3.access-key-id"] == "OLD"
    assert provider.generation == 0
    refresh.assert_not_called()


def test_provider_without_expiry_never_refreshes() -> None:
    refresh = MagicMock()
    provider = VendedCredentialsProvider({"s3.access-key-id": "OLD"}, expires_at_ms=None, refresh=refresh)
    provider.credentials()
    refresh.assert_not_called()


def test_pyarrow_fileio_rebuilds_filesystem_on_refresh() -> None:
    built: list[str | None] = []

    def fake_init_fs(self: PyArrowFileIO, scheme: str, netloc: str | None = None) -> str:
        ak = self.properties.get("s3.access-key-id")
        built.append(ak)
        return f"FS({ak})"

    with patch.object(PyArrowFileIO, "_initialize_fs", fake_init_fs):
        io = PyArrowFileIO({"s3.access-key-id": "OLD"})
        assert io.fs_by_scheme("s3") == "FS(OLD)"
        assert io.fs_by_scheme("s3") == "FS(OLD)"  # cached, _initialize_fs not called again
        assert io.fs_by_scheme.cache_info().hits == 1  # cache_info preserved (drop-in for lru_cache)

        io.set_credentials_provider(
            VendedCredentialsProvider(
                {"s3.access-key-id": "OLD"}, expires_at_ms=0, refresh=lambda: ({"s3.access-key-id": "NEW"}, None)
            )
        )
        assert io.fs_by_scheme("s3") == "FS(NEW)"  # near expiry -> refresh + rebuild
        assert io.properties["s3.access-key-id"] == "NEW"

    assert built == ["OLD", "NEW"]


def test_pyarrow_fileio_drops_provider_on_pickle() -> None:
    with patch.object(PyArrowFileIO, "_initialize_fs", lambda self, scheme, netloc=None: "FS"):
        io = PyArrowFileIO({"a": "b"})
        io.set_credentials_provider(VendedCredentialsProvider({}, expires_at_ms=0, refresh=lambda: ({}, None)))
        restored = pickle.loads(pickle.dumps(io))
        assert restored._credentials_provider is None


@pytest.mark.parametrize(
    "resolved, storage_credentials, expected",
    [
        ({"s3.session-token-expires-at-ms": "123"}, [], 123),  # standard, on the s3 credential
        ({}, [StorageCredential(prefix="oss", config={"fs.oss.token.expiration": "456"})], 456),  # DLF fallback
        ({"s3.access-key-id": "A"}, [StorageCredential(prefix="s3", config={"s3.access-key-id": "A"})], None),  # none
    ],
)
def test_extract_expires_at_ms_hybrid(resolved: dict, storage_credentials: list, expected: int | None) -> None:
    assert RestCatalog._extract_expires_at_ms(resolved, storage_credentials) == expected


def test_build_credentials_provider_reload_then_refresh() -> None:
    catalog = RestCatalog.__new__(RestCatalog)
    oss = StorageCredential(prefix="oss", config={"fs.oss.token.expiration": "1782428861000"})
    s3 = StorageCredential(prefix="s3", config={"s3.access-key-id": "A", "s3.session-token": "T"})
    table_response = MagicMock(storage_credentials=[oss, s3], config={})

    provider = catalog._build_credentials_provider(("ns", "t"), table_response, "oss://bucket/x", {"s3.access-key-id": "A"})
    assert provider is not None

    new_oss = StorageCredential(prefix="oss", config={"fs.oss.token.expiration": "9999999999999"})
    new_s3 = StorageCredential(prefix="s3", config={"s3.access-key-id": "NEW", "s3.session-token": "NT"})
    catalog._fetch_storage_credentials = MagicMock(return_value=[new_oss, new_s3])  # type: ignore[method-assign]
    provider._expires_at_ms = 0  # force near-expiry

    assert provider.credentials()["s3.access-key-id"] == "NEW"
    catalog._fetch_storage_credentials.assert_called_once_with(("ns", "t"), None)  # no endpoint -> reload path


def test_build_credentials_provider_prefers_endpoint() -> None:
    catalog = RestCatalog.__new__(RestCatalog)
    s3 = StorageCredential(prefix="s3", config={"s3.access-key-id": "A"})
    endpoint = "v1/p/namespaces/ns/tables/t/credentials"
    table_response = MagicMock(storage_credentials=[s3], config={"client.refresh-credentials-endpoint": endpoint})

    provider = catalog._build_credentials_provider(
        ("ns", "t"), table_response, "oss://bucket/x", {"s3.access-key-id": "A", "s3.session-token-expires-at-ms": "1000"}
    )
    assert provider is not None
    catalog._fetch_storage_credentials = MagicMock(return_value=[s3])  # type: ignore[method-assign]
    provider._expires_at_ms = 0
    provider.credentials()
    catalog._fetch_storage_credentials.assert_called_once_with(("ns", "t"), endpoint)


def test_build_credentials_provider_none_without_expiry() -> None:
    catalog = RestCatalog.__new__(RestCatalog)
    s3 = StorageCredential(prefix="s3", config={"s3.access-key-id": "A"})
    table_response = MagicMock(storage_credentials=[s3], config={})
    assert catalog._build_credentials_provider(("ns", "t"), table_response, "oss://b/x", {"s3.access-key-id": "A"}) is None


def test_provider_refresh_failure_within_buffer_keeps_old_creds() -> None:
    def boom() -> tuple[dict, int | None]:
        raise RuntimeError("transient network blip")

    # within the 5-min buffer but not yet expired -> failure is swallowed, old creds kept
    not_expired = int(time.time() * 1000) + 60_000
    provider = VendedCredentialsProvider({"s3.access-key-id": "OLD"}, not_expired, boom)
    assert provider.credentials()["s3.access-key-id"] == "OLD"


def test_provider_refresh_failure_after_expiry_reraises() -> None:
    def boom() -> tuple[dict, int | None]:
        raise RuntimeError("transient network blip")

    provider = VendedCredentialsProvider({"s3.access-key-id": "OLD"}, expires_at_ms=0, refresh=boom)
    with pytest.raises(RuntimeError):
        provider.credentials()


def test_provider_single_refresh_under_concurrency() -> None:
    calls = {"n": 0}

    def slow_refresh() -> tuple[dict, int | None]:
        calls["n"] += 1
        time.sleep(0.05)  # hold the lock so the other callers pile up
        return {"s3.access-key-id": "NEW"}, None  # no further expiry -> later callers don't refresh

    provider = VendedCredentialsProvider({"s3.access-key-id": "OLD"}, expires_at_ms=0, refresh=slow_refresh)
    results: list[str | None] = []

    def worker() -> None:
        results.append(provider.credentials()["s3.access-key-id"])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r == "NEW" for r in results)
    assert calls["n"] == 1  # exactly one refresh despite 8 concurrent callers
    assert provider.generation == 1
