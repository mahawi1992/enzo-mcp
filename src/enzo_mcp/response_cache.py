"""Local, bounded response memory for approved Jev requests.

The cache stores the provider response only. Selected context, evidence, approval
references, atom ids, and investigation ids never enter the database.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typesafe_sdk import SystemOneResponse

from .models import JevDispatchManifest

_CACHE_SCHEMA_VERSION = 1
_MAX_CACHED_RESPONSE_BYTES = 64 * 1024
_SAFE_SCOPE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MIN_TTL_SECONDS = 1
_MAX_TTL_SECONDS = 30 * 24 * 60 * 60
_MIN_ENTRIES = 1
_MAX_ENTRIES = 100_000


@dataclass(frozen=True, slots=True)
class JevCacheHit:
    """One validated cache hit and its non-sensitive cache identity."""

    response: SystemOneResponse
    cache_key: str
    namespace: str
    model_epoch: str
    response_sha256: str
    created_at: datetime
    expires_at: datetime


class JevResponseCache(Protocol):
    """Response-only cache used after manifest approval succeeds."""

    async def get(self, manifest: JevDispatchManifest) -> JevCacheHit | None:
        """Return a validated cached response for an exact manifest."""

    async def put(
        self,
        manifest: JevDispatchManifest,
        response: SystemOneResponse,
    ) -> None:
        """Store a validated provider response for an exact manifest."""


class SQLiteJevResponseCache:
    """A process-independent SQLite ledger of Jev response fingerprints."""

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        model_epoch: str,
        ttl_seconds: int = 86_400,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._namespace = self._validate_scope("namespace", namespace)
        self._model_epoch = self._validate_scope("model epoch", model_epoch)
        if not _MIN_TTL_SECONDS <= ttl_seconds <= _MAX_TTL_SECONDS:
            raise ValueError(
                f"cache TTL must be between {_MIN_TTL_SECONDS} and {_MAX_TTL_SECONDS} seconds"
            )
        if not _MIN_ENTRIES <= max_entries <= _MAX_ENTRIES:
            raise ValueError(f"cache max entries must be between {_MIN_ENTRIES} and {_MAX_ENTRIES}")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock

    @property
    def path(self) -> Path:
        """Expose the configured database path for diagnostics and tests."""

        return self._path

    @property
    def namespace(self) -> str:
        """Return the deployment-owned cache namespace."""

        return self._namespace

    @property
    def model_epoch(self) -> str:
        """Return the explicit model epoch that invalidates stale aliases."""

        return self._model_epoch

    def cache_key(self, manifest: JevDispatchManifest) -> str:
        """Bind cache identity to deployment scope and the exact logical request."""

        identity = "\0".join(
            (
                "enzo:jev:response",
                str(_CACHE_SCHEMA_VERSION),
                self._namespace,
                self._model_epoch,
                str(manifest.schema_version),
                manifest.canonical_sha256,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    async def get(self, manifest: JevDispatchManifest) -> JevCacheHit | None:
        """Read without blocking the MCP event loop."""

        return await asyncio.to_thread(self._get_sync, manifest)

    async def put(
        self,
        manifest: JevDispatchManifest,
        response: SystemOneResponse,
    ) -> None:
        """Write without blocking the MCP event loop."""

        await asyncio.to_thread(self._put_sync, manifest, response)

    def _get_sync(self, manifest: JevDispatchManifest) -> JevCacheHit | None:
        cache_key = self.cache_key(manifest)
        now = self._clock()
        connection = self._connect()
        try:
            with closing(connection), connection:
                row = connection.execute(
                    """
                    SELECT response_json, response_sha256, created_at, expires_at
                    FROM jev_responses
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
                if row is None:
                    return None
                response_json, response_sha256, created_at, expires_at = row
                if now >= float(expires_at):
                    connection.execute(
                        "DELETE FROM jev_responses WHERE cache_key = ?", (cache_key,)
                    )
                    return None
                response_bytes = str(response_json).encode("utf-8")
                if (
                    len(response_bytes) > _MAX_CACHED_RESPONSE_BYTES
                    or hashlib.sha256(response_bytes).hexdigest() != response_sha256
                ):
                    connection.execute(
                        "DELETE FROM jev_responses WHERE cache_key = ?", (cache_key,)
                    )
                    return None
                try:
                    response = SystemOneResponse.model_validate(json.loads(response_json))
                except (json.JSONDecodeError, TypeError, ValueError):
                    connection.execute(
                        "DELETE FROM jev_responses WHERE cache_key = ?", (cache_key,)
                    )
                    return None
                connection.execute(
                    "UPDATE jev_responses SET last_accessed_at = ? WHERE cache_key = ?",
                    (now, cache_key),
                )
        finally:
            self._harden_sidecars()
        return JevCacheHit(
            response=response,
            cache_key=cache_key,
            namespace=self._namespace,
            model_epoch=self._model_epoch,
            response_sha256=str(response_sha256),
            created_at=datetime.fromtimestamp(float(created_at), UTC),
            expires_at=datetime.fromtimestamp(float(expires_at), UTC),
        )

    def _put_sync(
        self,
        manifest: JevDispatchManifest,
        response: SystemOneResponse,
    ) -> None:
        response_json = json.dumps(
            response.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        response_bytes = response_json.encode("utf-8")
        if len(response_bytes) > _MAX_CACHED_RESPONSE_BYTES:
            raise ValueError("Jev response exceeds the 64KiB cache record limit")
        now = self._clock()
        cache_key = self.cache_key(manifest)
        connection = self._connect()
        try:
            with closing(connection), connection:
                connection.execute("DELETE FROM jev_responses WHERE expires_at <= ?", (now,))
                connection.execute(
                    """
                    INSERT INTO jev_responses (
                        cache_key,
                        namespace,
                        model_epoch,
                        manifest_schema_version,
                        manifest_sha256,
                        response_json,
                        response_sha256,
                        created_at,
                        expires_at,
                        last_accessed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        response_json = excluded.response_json,
                        response_sha256 = excluded.response_sha256,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        last_accessed_at = excluded.last_accessed_at
                    """,
                    (
                        cache_key,
                        self._namespace,
                        self._model_epoch,
                        manifest.schema_version,
                        manifest.canonical_sha256,
                        response_json,
                        hashlib.sha256(response_bytes).hexdigest(),
                        now,
                        now + self._ttl_seconds,
                        now,
                    ),
                )
                count = int(connection.execute("SELECT COUNT(*) FROM jev_responses").fetchone()[0])
                excess = count - self._max_entries
                if excess > 0:
                    connection.execute(
                        """
                        DELETE FROM jev_responses
                        WHERE cache_key IN (
                            SELECT cache_key
                            FROM jev_responses
                            ORDER BY last_accessed_at ASC, created_at ASC, cache_key ASC
                            LIMIT ?
                        )
                        """,
                        (excess,),
                    )
        finally:
            self._harden_sidecars()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.parent.is_symlink() or not self._path.parent.is_dir():
            raise OSError("Jev cache parent must be a real directory")
        self._path.parent.chmod(0o700)
        if self._path.is_symlink():
            raise OSError("Jev cache database must not be a symbolic link")
        if self._path.exists():
            if not self._path.is_file():
                raise OSError("Jev cache database must be a regular file")
            self._path.chmod(0o600)
        connection = sqlite3.connect(self._path, timeout=5)
        try:
            self._path.chmod(0o600)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jev_responses (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    model_epoch TEXT NOT NULL,
                    manifest_schema_version INTEGER NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    response_sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL
                )
                """
            )
            self._harden_sidecars()
        except Exception:
            connection.close()
            raise
        return connection

    def _harden_sidecars(self) -> None:
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self._path}{suffix}")
            if sidecar.is_symlink():
                raise OSError("Jev cache sidecar must not be a symbolic link")
            if not sidecar.exists():
                continue
            if not sidecar.is_file():
                raise OSError("Jev cache sidecar must be a regular file")
            sidecar.chmod(0o600)

    @staticmethod
    def _validate_scope(name: str, value: str) -> str:
        normalized = value.strip()
        if not _SAFE_SCOPE.fullmatch(normalized):
            raise ValueError(
                f"cache {name} must contain 1-128 letters, digits, dots, dashes, or underscores"
            )
        return normalized


def response_cache_from_env(
    environment: Mapping[str, str] | None = None,
) -> SQLiteJevResponseCache | None:
    """Build the optional cache from deployment-owned environment settings."""

    values = os.environ if environment is None else environment
    cache_dir = values.get("ENZO_JEV_CACHE_DIR", "").strip()
    if not cache_dir:
        return None
    namespace = values.get("ENZO_JEV_CACHE_NAMESPACE", "").strip()
    model_epoch = values.get("ENZO_JEV_CACHE_MODEL_EPOCH", "").strip()
    if not namespace or not model_epoch:
        raise ValueError(
            "ENZO_JEV_CACHE_NAMESPACE and ENZO_JEV_CACHE_MODEL_EPOCH are required when "
            "ENZO_JEV_CACHE_DIR is set"
        )
    try:
        ttl_seconds = int(values.get("ENZO_JEV_CACHE_TTL_SECONDS", "86400"))
        max_entries = int(values.get("ENZO_JEV_CACHE_MAX_ENTRIES", "10000"))
    except ValueError as error:
        raise ValueError("Jev cache TTL and max entries must be integers") from error
    return SQLiteJevResponseCache(
        Path(cache_dir).expanduser() / "jev-responses.sqlite3",
        namespace=namespace,
        model_epoch=model_epoch,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
    )
