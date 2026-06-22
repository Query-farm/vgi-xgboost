"""In-process VGI invocation helpers shared across the worker test suite.

These drive a function through the real bind -> init -> process (-> finalize)
lifecycle without spawning a worker process, so tests stay fast and debuggable.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from vgi.arguments import Arguments
from vgi.function_storage import BoundStorage, FunctionStorage, FunctionStorageSqlite
from vgi.invocation import FunctionType
from vgi.protocol import BindRequest, InitRequest
from vgi.table_function import ProcessParams


def test_storage() -> FunctionStorage:
    """Real in-memory FunctionStorage for the function lifecycle in tests."""
    return FunctionStorageSqlite(":memory:")


class MockOutputCollector:
    """Captures emitted batches (and partition values) for assertions."""

    def __init__(self, output_schema: pa.Schema) -> None:
        self.output_schema = output_schema
        self.batches: list[pa.RecordBatch] = []
        self.partition_values: list[dict[str, Any] | None] = []
        self._finished = False

    def emit(
        self,
        batch: pa.RecordBatch,
        partition_values: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.batches.append(batch)
        self.partition_values.append(partition_values)

    def finish(self) -> None:
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def emit_pydict(self, data: dict[str, Any], schema: pa.Schema | None = None) -> None:
        self.emit(pa.RecordBatch.from_pydict(data, schema=schema or self.output_schema))

    def emit_client_log_message(self, msg: Any) -> None:
        pass


def invoke_table_function(
    func_cls: type,
    *,
    named: dict[str, pa.Scalar] | None = None,
    positional: tuple[pa.Scalar, ...] = (),
) -> pa.Table:
    """Run a (source) table function through bind -> init -> process -> table."""
    args = Arguments(positional=positional, named=named or {})

    bind_req = BindRequest(
        function_name=func_cls.Meta.name,
        arguments=args,
        function_type=FunctionType.TABLE,
    )
    bind_resp = func_cls.bind(bind_req)

    init_req = InitRequest(bind_call=bind_req, output_schema=bind_resp.output_schema)
    init_resp = func_cls.global_init(init_req)

    storage = test_storage()
    params = ProcessParams(
        args=func_cls._parse_arguments(func_cls.FunctionArguments, args),
        init_call=init_req,
        init_response=init_resp,
        output_schema=bind_resp.output_schema,
        settings={},
        secrets={},
        storage=BoundStorage(storage, init_resp.execution_id),
    )

    state = func_cls.initial_state(params)
    out = MockOutputCollector(bind_resp.output_schema)

    while not out.finished:
        func_cls.process(params, state, out)

    return pa.Table.from_batches(out.batches, schema=bind_resp.output_schema)
