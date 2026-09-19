from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from stat import S_IMODE
from typing import cast

from typesafe_sdk import (
    Answer,
    JSONContent,
    NoulAnswer,
    Questions,
    RetryPolicy,
    SystemOneResponse,
    Usage,
)

from enzo_mcp.engine import EnzoEngine
from enzo_mcp.models import (
    AtomizeRequest,
    ContextItem,
    ContextKind,
    EvidenceKind,
    EvidenceRequirement,
    ExpectedAnswerType,
    JevContextState,
    JevDispatchManifest,
    JevDispatchSelection,
    JevLogicalRequest,
    JevNoulQuestion,
    JevQuestion,
    JevQuestions,
    JEVResult,
    JevState,
    ObserveRequest,
    PredicateOperator,
    Quantifier,
    SemanticStatus,
    VerificationMethod,
)
from enzo_mcp.response_cache import (
    JevCacheHit,
    JevResponseCache,
    SQLiteJevResponseCache,
    response_cache_from_env,
)
from enzo_mcp.sensor import PydanticJevSensor


class RecordingClient:
    def __init__(self, answer: Answer | None = None) -> None:
        self.answer = answer if answer is not None else NoulAnswer(noul=0.95)
        self.calls = 0

    async def system_one(
        self,
        state: JSONContent,
        questions: Questions,
        *,
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> SystemOneResponse:
        del state, questions, model, retry
        self.calls += 1
        return _response(self.answer)


class MemoryResponseCache:
    def __init__(self) -> None:
        self.values: dict[str, SystemOneResponse] = {}
        self.gets = 0
        self.puts = 0

    async def get(self, manifest: JevDispatchManifest) -> JevCacheHit | None:
        self.gets += 1
        response = self.values.get(manifest.canonical_sha256)
        if response is None:
            return None
        return JevCacheHit(
            response=response,
            cache_key=manifest.canonical_sha256,
            namespace="tests",
            model_epoch="jev-test",
            response_sha256=hashlib.sha256(
                json.dumps(response.model_dump(mode="json"), sort_keys=True).encode()
            ).hexdigest(),
            created_at=datetime(2026, 9, 19, tzinfo=UTC),
            expires_at=datetime(2026, 9, 20, tzinfo=UTC),
        )

    async def put(
        self,
        manifest: JevDispatchManifest,
        response: SystemOneResponse,
    ) -> None:
        self.puts += 1
        self.values[manifest.canonical_sha256] = response


class ReadFailingCache(MemoryResponseCache):
    async def get(self, manifest: JevDispatchManifest) -> JevCacheHit | None:
        del manifest
        self.gets += 1
        raise OSError("simulated cache read failure")


class WriteFailingCache(MemoryResponseCache):
    async def put(
        self,
        manifest: JevDispatchManifest,
        response: SystemOneResponse,
    ) -> None:
        del manifest, response
        self.puts += 1
        raise OSError("simulated cache write failure")


def _response(answer: Answer | None = None) -> SystemOneResponse:
    return SystemOneResponse(
        model="jev-1.13.0",
        usage=Usage(input_tokens=12, output_tokens=1),
        answers={"claim": answer if answer is not None else NoulAnswer(noul=0.95)},
    )


def _manifest(seed: str, *, secret: str = "SECRET-CONTEXT-CANARY") -> JevDispatchManifest:
    logical_request = JevLogicalRequest(
        model="jev-latest",
        state=JevState(
            claim=JevQuestion(
                question="Is the deployment safe?",
                subject="the deployment",
                predicate="is safe",
                scope="test",
                operator=PredicateOperator.IS_TRUE,
                quantifier=Quantifier.ONE,
                expected_answer_type=ExpectedAnswerType.BOOLEAN,
            ),
            scope="test",
            context=(
                JevContextState(
                    key="selected",
                    value=secret,
                    kind=ContextKind.ASSUMPTION,
                ),
            ),
        ),
        questions=JevQuestions(claim=JevNoulQuestion(instructions="Is the deployment safe?")),
    )
    return JevDispatchManifest(
        canonical_sha256=hashlib.sha256(seed.encode()).hexdigest(),
        byte_length=len(json.dumps(logical_request.model_dump(mode="json"))),
        logical_request=logical_request,
        selection=JevDispatchSelection(context_keys=("selected",)),
    )


def _semantic_requirement() -> EvidenceRequirement:
    return EvidenceRequirement(
        id="jev-observation",
        description="Jev evaluates the caller-selected state",
        accepted_kinds=(EvidenceKind.SENSOR_OUTPUT,),
    )


async def _preview_and_approve(
    engine: EnzoEngine,
    make_request: Callable[..., AtomizeRequest],
    *,
    atom_id: str,
    approval_reference: str,
) -> tuple[JEVResult, JevDispatchManifest]:
    atom = (
        await engine.atomize(
            make_request(
                atom_id=atom_id,
                context=(
                    ContextItem(
                        key="selected",
                        value={"safe": True},
                        kind=ContextKind.ASSUMPTION,
                    ),
                ),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))
    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approval_reference=approval_reference,
        )
    )
    assert preview.dispatch_manifest is not None
    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
            approval_reference=approval_reference,
        )
    )
    return result, preview.dispatch_manifest


async def test_sqlite_cache_round_trip_omits_request_state(tmp_path: Path) -> None:
    database = tmp_path / "private" / "responses.sqlite3"
    cache = SQLiteJevResponseCache(
        database,
        namespace="project-a",
        model_epoch="jev-1.13.0",
    )
    manifest = _manifest("round-trip")

    await cache.put(manifest, _response())
    hit = await cache.get(manifest)

    assert hit is not None
    assert hit.response == _response()
    assert hit.namespace == "project-a"
    assert b"SECRET-CONTEXT-CANARY" not in database.read_bytes()
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT response_json, manifest_sha256 FROM jev_responses"
        ).fetchone()
    assert stored is not None
    assert "SECRET-CONTEXT-CANARY" not in stored[0]
    assert stored[1] == manifest.canonical_sha256


async def test_cache_hardens_existing_paths_and_wal_sidecars(tmp_path: Path) -> None:
    directory = tmp_path / "permissive"
    directory.mkdir(mode=0o777)
    database = directory / "responses.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE seed (value INTEGER)")
    directory.chmod(0o777)
    database.chmod(0o666)
    cache = SQLiteJevResponseCache(
        database,
        namespace="tests",
        model_epoch="jev-1.13.0",
    )

    connection = cache._connect()
    try:
        sidecars = tuple(
            path for path in (Path(f"{database}-wal"), Path(f"{database}-shm")) if path.exists()
        )
        assert sidecars
        assert all(S_IMODE(path.stat().st_mode) == 0o600 for path in sidecars)
    finally:
        connection.close()

    assert S_IMODE(directory.stat().st_mode) == 0o700
    assert S_IMODE(database.stat().st_mode) == 0o600


async def test_sqlite_cache_scope_and_epoch_are_part_of_identity(tmp_path: Path) -> None:
    database = tmp_path / "responses.sqlite3"
    manifest = _manifest("scoped")
    original = SQLiteJevResponseCache(
        database,
        namespace="project-a",
        model_epoch="jev-1.13.0",
    )
    other_namespace = SQLiteJevResponseCache(
        database,
        namespace="project-b",
        model_epoch="jev-1.13.0",
    )
    other_epoch = SQLiteJevResponseCache(
        database,
        namespace="project-a",
        model_epoch="jev-1.14.0",
    )

    await original.put(manifest, _response())

    assert await original.get(manifest) is not None
    assert await other_namespace.get(manifest) is None
    assert await other_epoch.get(manifest) is None
    assert original.cache_key(manifest) != other_namespace.cache_key(manifest)
    assert original.cache_key(manifest) != other_epoch.cache_key(manifest)
    assert original.cache_key(manifest) != original.cache_key(_manifest("changed-request"))


async def test_sqlite_cache_expires_entries(tmp_path: Path) -> None:
    now = [100.0]
    cache = SQLiteJevResponseCache(
        tmp_path / "responses.sqlite3",
        namespace="tests",
        model_epoch="jev-1.13.0",
        ttl_seconds=5,
        clock=lambda: now[0],
    )
    manifest = _manifest("expiry")
    await cache.put(manifest, _response())
    assert await cache.get(manifest) is not None

    now[0] = 105.0

    assert await cache.get(manifest) is None


async def test_sqlite_cache_evicts_oldest_entry(tmp_path: Path) -> None:
    now = [100.0]
    cache = SQLiteJevResponseCache(
        tmp_path / "responses.sqlite3",
        namespace="tests",
        model_epoch="jev-1.13.0",
        max_entries=2,
        clock=lambda: now[0],
    )
    manifests = tuple(_manifest(f"entry-{index}") for index in range(3))
    for manifest in manifests:
        await cache.put(manifest, _response())
        now[0] += 1

    assert await cache.get(manifests[0]) is None
    assert await cache.get(manifests[1]) is not None
    assert await cache.get(manifests[2]) is not None


async def test_sqlite_cache_discards_corrupt_entry(tmp_path: Path) -> None:
    database = tmp_path / "responses.sqlite3"
    cache = SQLiteJevResponseCache(
        database,
        namespace="tests",
        model_epoch="jev-1.13.0",
    )
    manifest = _manifest("corrupt")
    await cache.put(manifest, _response())
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE jev_responses SET response_json = ?",
            ('{"answers":{"claim":{"type":"noul","noul":0.99}}}',),
        )

    assert await cache.get(manifest) is None


def test_cache_environment_requires_scope_and_bounds(tmp_path: Path) -> None:
    assert response_cache_from_env({}) is None
    cache = response_cache_from_env(
        {
            "ENZO_JEV_CACHE_DIR": str(tmp_path),
            "ENZO_JEV_CACHE_NAMESPACE": "project-a",
            "ENZO_JEV_CACHE_MODEL_EPOCH": "jev-1.13.0",
            "ENZO_JEV_CACHE_TTL_SECONDS": "60",
            "ENZO_JEV_CACHE_MAX_ENTRIES": "25",
        }
    )
    assert cache is not None
    assert cache.path == tmp_path / "jev-responses.sqlite3"


async def test_approved_identical_request_reuses_cached_typed_response(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    cache = MemoryResponseCache()
    client = RecordingClient()
    sensor = PydanticJevSensor(client=client, cache=cache)

    first, first_manifest = await _preview_and_approve(
        EnzoEngine(sensor=sensor),
        make_request,
        atom_id="first-cache-atom",
        approval_reference="first-approval",
    )
    second, second_manifest = await _preview_and_approve(
        EnzoEngine(sensor=sensor),
        make_request,
        atom_id="second-cache-atom",
        approval_reference="second-approval",
    )

    assert first_manifest.canonical_sha256 == second_manifest.canonical_sha256
    assert first.status is SemanticStatus.VERIFIED
    assert second.status is SemanticStatus.VERIFIED
    assert client.calls == 1
    assert cache.puts == 1
    second_evidence = second.evidence[-1]
    second_payload = cast(dict[str, object], second_evidence.payload)
    cache_payload = cast(dict[str, object], second_payload["cache"])
    assert cache_payload["hit"] is True
    assert cache_payload["created_at"] == "2026-09-19T00:00:00+00:00"
    assert cache_payload["expires_at"] == "2026-09-20T00:00:00+00:00"
    assert second_evidence.provenance.observed_at == datetime(2026, 9, 19, tzinfo=UTC)
    assert second_evidence.provenance.content_hash == cache_payload["response_sha256"]
    assert second_evidence.provenance.provider == "enzo/jev-response-cache"


async def test_unauthorized_paths_do_not_read_cache(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    cache = MemoryResponseCache()
    client = RecordingClient()
    engine = EnzoEngine(sensor=PydanticJevSensor(client=client, cache=cache))
    atom = (
        await engine.atomize(
            make_request(
                atom_id="authorization-cache-atom",
                context=(
                    ContextItem(
                        key="selected",
                        value=True,
                        kind=ContextKind.ASSUMPTION,
                    ),
                ),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))

    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
        )
    )
    assert preview.dispatch_manifest is not None
    await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256="0" * 64,
        )
    )

    assert cache.gets == 0
    assert cache.puts == 0
    assert client.calls == 0


async def test_cache_read_and_write_failures_preserve_provider_semantics(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    read_client = RecordingClient()
    read_result, _ = await _preview_and_approve(
        EnzoEngine(sensor=PydanticJevSensor(client=read_client, cache=ReadFailingCache())),
        make_request,
        atom_id="read-failure-atom",
        approval_reference="read-failure",
    )
    write_client = RecordingClient()
    write_cache = WriteFailingCache()
    write_result, _ = await _preview_and_approve(
        EnzoEngine(sensor=PydanticJevSensor(client=write_client, cache=write_cache)),
        make_request,
        atom_id="write-failure-atom",
        approval_reference="write-failure",
    )

    assert read_result.status is SemanticStatus.VERIFIED
    assert write_result.status is SemanticStatus.VERIFIED
    assert read_client.calls == 1
    assert write_client.calls == 1
    assert write_cache.puts == 1


async def test_invalid_cached_answer_falls_through_to_provider(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    cache = MemoryResponseCache()
    client = RecordingClient()
    sensor = PydanticJevSensor(client=client, cache=cache)
    engine = EnzoEngine(sensor=sensor)
    atom = (
        await engine.atomize(
            make_request(
                atom_id="invalid-cache-answer",
                context=(
                    ContextItem(
                        key="selected",
                        value=True,
                        kind=ContextKind.ASSUMPTION,
                    ),
                ),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))
    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
        )
    )
    assert preview.dispatch_manifest is not None
    cache.values[preview.dispatch_manifest.canonical_sha256] = SystemOneResponse(
        model="jev-1.13.0",
        usage=Usage(input_tokens=1, output_tokens=1),
        answers={},
    )

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
        )
    )

    assert result.status is SemanticStatus.VERIFIED
    assert client.calls == 1
    assert cache.puts == 1


async def test_cache_hit_can_be_used_without_provider_credentials(
    make_request: Callable[..., AtomizeRequest],
) -> None:
    cache = MemoryResponseCache()
    engine = EnzoEngine(sensor=PydanticJevSensor(cache=cache))
    atom = (
        await engine.atomize(
            make_request(
                atom_id="offline-cache-hit",
                context=(
                    ContextItem(
                        key="selected",
                        value=True,
                        kind=ContextKind.ASSUMPTION,
                    ),
                ),
                evidence_requirements=(_semantic_requirement(),),
                verification_method=VerificationMethod.SEMANTIC_SENSOR,
            )
        )
    ).atom
    selection = JevDispatchSelection(context_keys=("selected",))
    preview = await engine.observe(
        ObserveRequest(
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
        )
    )
    assert preview.dispatch_manifest is not None
    cache.values[preview.dispatch_manifest.canonical_sha256] = _response()

    result = await engine.observe(
        ObserveRequest(
            allow_external_jev=True,
            investigation_id=atom.investigation_id,
            atom_id=atom.id,
            dispatch_selection=selection,
            approved_dispatch_sha256=preview.dispatch_manifest.canonical_sha256,
        )
    )

    assert result.status is SemanticStatus.VERIFIED
    payload = cast(dict[str, object], result.evidence[-1].payload)
    cache_payload = cast(dict[str, object], payload["cache"])
    assert cache_payload["hit"] is True


def test_memory_cache_implements_protocol() -> None:
    cache: JevResponseCache = MemoryResponseCache()
    assert cache is not None
