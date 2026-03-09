# -*- coding: utf-8 -*-
"""The main entry point of the ReAct agent example."""
import asyncio
import datetime
import inspect
import json
import os
import sys
import tempfile
from typing import Any

import shortuuid
from dotenv import load_dotenv

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse


def _parse_json_if_needed(value: Any) -> Any:
    """Parse JSON string input if needed."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _extract_tool_response_text(response: ToolResponse) -> str:
    """Extract plain text from a tool response for JSON transport."""

    texts: list[str] = []
    for block in response.content:
        if block.get("type") == "text" and block.get("text"):
            texts.append(block["text"])
    return "\n".join(texts)


def _ensure_tool_response(value: Any) -> ToolResponse:
    """Ensure a value can be represented as a ``ToolResponse``."""

    if isinstance(value, ToolResponse):
        return value

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    "Unexpected tool result type: "
                    f"{type(value).__name__}."
                ),
            ),
        ],
    )


def _now_iso() -> str:
    """Return current local datetime in ISO format."""

    return datetime.datetime.now().isoformat()

# simple in-memory structure for tracking spawned tasks when blocking=False
# each entry maps task_id -> metadata dict containing status, timestamps and
# the underlying asyncio.Task object.  This is intentionally minimal; in a
# production environment you'd want persistence, locks, cancellation support,
# etc.
_tasks: dict[str, dict] = {}
_subscription_result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

async def execute_python_code(
    code: str,
    timeout: float = 300,
    blocking: bool = True,
    subscribe: bool = False,
    **kwargs: Any,
) -> ToolResponse:
    """Execute Python code, optionally in a background task.

    Args:
        code (`str`):
            The Python code to be executed.
        timeout (`float`, defaults to `300`):
            The maximum time (in seconds) allowed for the code to run.
        blocking (`bool`, defaults to `True`):
            If ``True``, wait for execution to finish and return the result.
            If ``False``, schedule execution in background and return task
            metadata immediately.
        subscribe (`bool`, defaults to `False`):
            Whether to emit completion events for ``wait_task_result`` when
            ``blocking=False``.

    Returns:
        `ToolResponse`:
            The execution result when blocking, or task metadata when
            non-blocking.
    """

    parsed_blocking = _parse_json_if_needed(blocking)
    parsed_subscribe = _parse_json_if_needed(subscribe)

    if not isinstance(parsed_blocking, bool):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid blocking format: expected bool.",
                ),
            ],
        )

    if not isinstance(parsed_subscribe, bool):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid subscribe format: expected bool.",
                ),
            ],
        )

    async def _execute_once() -> ToolResponse:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, f"tmp_{shortuuid.uuid()}.py")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
                stdout, stderr = await proc.communicate()
                stdout_str = stdout.decode("utf-8")
                stderr_str = stderr.decode("utf-8")
                returncode = proc.returncode

            except asyncio.TimeoutError:
                stderr_suffix = (
                    f"TimeoutError: The code execution exceeded "
                    f"the timeout of {timeout} seconds."
                )
                returncode = -1
                try:
                    proc.terminate()
                    stdout, stderr = await proc.communicate()
                    stdout_str = stdout.decode("utf-8")
                    stderr_str = stderr.decode("utf-8")
                    if stderr_str:
                        stderr_str += f"\n{stderr_suffix}"
                    else:
                        stderr_str = stderr_suffix
                except ProcessLookupError:
                    stdout_str = ""
                    stderr_str = stderr_suffix

            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"<returncode>{returncode}</returncode>"
                        f"<stdout>{stdout_str}</stdout>"
                        f"<stderr>{stderr_str}</stderr>",
                    ),
                ],
            )

    if parsed_blocking:
        return await _execute_once()

    task_id = shortuuid.uuid()
    created_at = _now_iso()
    metadata: dict[str, Any] = {
        "status": "queued",
        "created_at": created_at,
        "start_iso_time": created_at,
        "last_received_iso_time": None,
        "message": "Task started successfully.",
        "future": None,
        "subscribe": parsed_subscribe,
    }

    async def _wrapper() -> ToolResponse:
        metadata["status"] = "running"
        try:
            result = await _execute_once()
            metadata["status"] = "completed"
            metadata["result"] = result

            if metadata["subscribe"]:
                last_received_iso_time = _now_iso()
                metadata["last_received_iso_time"] = last_received_iso_time
                await _subscription_result_queue.put(
                    {
                        "task_id": task_id,
                        "name": "execute_python_code",
                        "status": "completed",
                        "result": _extract_tool_response_text(result),
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": last_received_iso_time,
                    }
                )

            return result
        except Exception as exc:  # pragma: no cover - example code
            metadata["status"] = "failed"
            metadata["message"] = str(exc)

            if metadata["subscribe"]:
                last_received_iso_time = _now_iso()
                metadata["last_received_iso_time"] = last_received_iso_time
                await _subscription_result_queue.put(
                    {
                        "task_id": task_id,
                        "name": "execute_python_code",
                        "status": "failed",
                        "error": str(exc),
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": last_received_iso_time,
                    }
                )

            raise

    fut = asyncio.create_task(_wrapper())
    metadata["future"] = fut
    _tasks[task_id] = metadata

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "task_id": task_id,
                        "name": "execute_python_code",
                        "status": metadata["status"],
                        "created_at": created_at,
                        "message": metadata["message"],
                        "subscribe": metadata["subscribe"],
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": metadata[
                            "last_received_iso_time"
                        ],
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )

async def subscribe_system_time(
    interval_seconds: float = 4,
    **kwargs: Any,
) -> ToolResponse:
    """Subscribe system time updates and publish them to the queue.

    Args:
        interval_seconds (`float`, defaults to `4`):
            Interval in seconds between two system time updates.

    Returns:
        `ToolResponse`:
            Task metadata for the background subscription publisher.
    """

    parsed_interval = _parse_json_if_needed(interval_seconds)
    if not isinstance(parsed_interval, (int, float)):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid interval_seconds format: expected number.",
                ),
            ],
        )

    interval = float(parsed_interval)
    if interval <= 0:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="interval_seconds must be greater than 0.",
                ),
            ],
        )

    task_id = shortuuid.uuid()
    created_at = _now_iso()
    metadata: dict[str, Any] = {
        "status": "queued",
        "created_at": created_at,
        "start_iso_time": created_at,
        "last_received_iso_time": None,
        "message": "System time subscription started.",
        "future": None,
        "subscribe": True,
    }

    async def _publisher() -> None:
        metadata["status"] = "running"
        try:
            while True:
                await asyncio.sleep(interval)
                last_received_iso_time = _now_iso()
                metadata["last_received_iso_time"] = last_received_iso_time

                await _subscription_result_queue.put(
                    {
                        "task_id": task_id,
                        "name": "subscribe_system_time",
                        "status": "streaming",
                        "result": last_received_iso_time,
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": last_received_iso_time,
                        "blocking": False,
                    }
                )
        except asyncio.CancelledError:  # pragma: no cover - example code
            metadata["status"] = "cancelled"
            metadata["message"] = "System time subscription cancelled."
            raise
        except Exception as exc:  # pragma: no cover - example code
            metadata["status"] = "failed"
            metadata["message"] = str(exc)
            last_received_iso_time = _now_iso()
            metadata["last_received_iso_time"] = last_received_iso_time
            await _subscription_result_queue.put(
                {
                    "task_id": task_id,
                    "name": "subscribe_system_time",
                    "status": "failed",
                    "error": str(exc),
                    "start_iso_time": metadata["start_iso_time"],
                    "last_received_iso_time": last_received_iso_time,
                }
            )
            raise

    fut = asyncio.create_task(_publisher())
    metadata["future"] = fut
    _tasks[task_id] = metadata

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "task_id": task_id,
                        "name": "subscribe_system_time",
                        "status": metadata["status"],
                        "created_at": created_at,
                        "message": metadata["message"],
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": metadata[
                            "last_received_iso_time"
                        ],
                        "blocking": False,
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )

async def wait_task_result(
    task_ids: list[str] | str | None = None,
) -> ToolResponse:
    """Wait for background tasks to complete.

    Args:
        task_ids (`list[str] | str | None`, optional):
            Task ids to wait for. A JSON encoded list string is also
            supported. If omitted or empty, this call waits for one
            subscribed result and returns it.

    Returns:
        `ToolResponse`:
            A JSON summary of waited task results, or one subscribed result.
    """

    normalized_ids: list[str] = []

    parsed_task_ids = _parse_json_if_needed(task_ids)

    async def _wait_subscription_results() -> ToolResponse:
        buffered_results: list[dict[str, Any]] = []

        while True:
            subscription_result = await _subscription_result_queue.get()
            start_iso_time = subscription_result.get("start_iso_time")
            last_received_iso_time = _now_iso()

            subscription_result["start_iso_time"] = start_iso_time
            subscription_result["last_received_iso_time"] = (
                last_received_iso_time
            )

            task_id = subscription_result.get("task_id")
            if isinstance(task_id, str) and task_id in _tasks:
                _tasks[task_id]["last_received_iso_time"] = (
                    last_received_iso_time
                )

            buffered_results.append(subscription_result)

            # Continue accumulating queue items for stream-like updates.
            if subscription_result.get("blocking") is False:
                continue

            if len(buffered_results) == 1:
                response_payload: dict[str, Any] | list[dict[str, Any]] = (
                    buffered_results[0]
                )
            else:
                response_payload = {"results": buffered_results}

            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=json.dumps(response_payload, ensure_ascii=False),
                    ),
                ],
            )

    # Subscription mode: when no task ids are provided, block until any
    # subscribed task result arrives, then return exactly one result.
    if parsed_task_ids is None or parsed_task_ids == "" or parsed_task_ids == []:
        return await _wait_subscription_results()

    if isinstance(parsed_task_ids, str):
        normalized_ids.append(parsed_task_ids)
    elif isinstance(parsed_task_ids, list):
        normalized_ids.extend(str(item) for item in parsed_task_ids)
    else:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid task_ids format: expected list[str] or JSON string.",
                ),
            ],
        )

    # Keep insertion order while removing duplicated task IDs.
    deduped_ids: list[str] = []
    seen: set[str] = set()
    for item in normalized_ids:
        if item and item not in seen:
            seen.add(item)
            deduped_ids.append(item)

    if not deduped_ids:
        return await _wait_subscription_results()

    async def _wait_one(
        one_task_id: str,
    ) -> dict[str, Any]:
        info = _tasks.get(one_task_id)
        start_iso_time: str | None = None
        if info is not None:
            start_iso_time = info.get("start_iso_time") or info.get("created_at")

        if info is None:
            return {
                "task_id": one_task_id,
                "result": None,
                "error": f"Task {one_task_id} not found.",
                "start_iso_time": None,
                "last_received_iso_time": _now_iso(),
            }

        fut = info.get("future")
        if fut is None:
            return {
                "task_id": one_task_id,
                "result": None,
                "error": "No associated future for task.",
                "start_iso_time": start_iso_time,
                "last_received_iso_time": _now_iso(),
            }

        try:
            result = await fut
            last_received_iso_time = _now_iso()
            info["last_received_iso_time"] = last_received_iso_time
            return {
                "task_id": one_task_id,
                "result": result,
                "error": None,
                "start_iso_time": start_iso_time,
                "last_received_iso_time": last_received_iso_time,
            }
        except Exception as exc:
            last_received_iso_time = _now_iso()
            info["last_received_iso_time"] = last_received_iso_time
            return {
                "task_id": one_task_id,
                "result": None,
                "error": f"Task failed: {exc}",
                "start_iso_time": start_iso_time,
                "last_received_iso_time": last_received_iso_time,
            }

    waited_results = await asyncio.gather(
        *[_wait_one(one_task_id) for one_task_id in deduped_ids],
    )

    payload: list[dict[str, Any]] = []
    for waited in waited_results:
        one_task_id = waited["task_id"]
        result = waited["result"]
        err = waited["error"]
        start_iso_time = waited.get("start_iso_time")
        last_received_iso_time = waited.get("last_received_iso_time")

        if err is not None:
            payload.append(
                {
                    "task_id": one_task_id,
                    "status": "failed",
                    "error": err,
                    "start_iso_time": start_iso_time,
                    "last_received_iso_time": last_received_iso_time,
                }
            )
            continue

        normalized_result = (
            _ensure_tool_response(result)
            if result is not None
            else ToolResponse(content=[])
        )

        payload.append(
            {
                "task_id": one_task_id,
                "status": "completed",
                "result": _extract_tool_response_text(
                    normalized_result,
                ),
                "start_iso_time": start_iso_time,
                "last_received_iso_time": last_received_iso_time,
            }
        )

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps({"results": payload}, ensure_ascii=False),
            ),
        ],
    )

async def main() -> None:
    """The main entry point for the ReAct agent example."""
    load_dotenv()
    toolkit = Toolkit()

    # ``batch_run_tool`` is a thin wrapper around the toolkit itself. It
    # accepts a batch ``calls`` payload and returns merged results.
    async def batch_run_tool(
        calls: list[dict[str, Any]] | str,
    ) -> ToolResponse:
        """Invoke registered tools in a single batch call.

        Args:
            calls (`list[dict[str, Any]] | str`): batch-call payload. Each
                item should contain ``name`` and optional ``input``.
        """

        # helper that actually executes the tool and accumulates the final
        # chunk as a single ``ToolResponse`` so callers get a normal object
        async def _execute_once(
            call_name: str,
            call_input: dict | str | None,
        ) -> ToolResponse:
            # The agent sometimes serializes the input dict as a JSON string,
            # so try to decode it back to a mapping for convenience.
            parsed_input = _parse_json_if_needed(call_input)

            call = {"name": call_name, "input": parsed_input}
            final: ToolResponse | None = None

            # `call_tool_function` returns a coroutine that yields an
            # async-generator. Always await first, then consume chunks.
            res_gen = await toolkit.call_tool_function(call)
            if inspect.isasyncgen(res_gen):
                async for chunk in res_gen:
                    final = _ensure_tool_response(chunk)
            else:
                final = _ensure_tool_response(res_gen)

            return final or ToolResponse(content=[])

        parsed_calls = _parse_json_if_needed(calls)
        pending_calls: list[tuple[str, dict | str | None]] = []

        if not isinstance(parsed_calls, list):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Invalid calls format: expected list[dict].",
                    ),
                ],
            )

        for index, call_item in enumerate(parsed_calls):
            if not isinstance(call_item, dict):
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=f"Invalid call item at index {index}: expected dict.",
                        ),
                    ],
                )

            call_name = call_item.get("name")
            if not isinstance(call_name, str) or not call_name:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=f"Invalid call item at index {index}: missing name.",
                        ),
                    ],
                )

            pending_calls.append((call_name, call_item.get("input")))

        if not pending_calls:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="calls must contain at least one tool call.",
                    ),
                ],
            )

        call_results = await asyncio.gather(
            *[
                _execute_once(call_name, call_input)
                for call_name, call_input in pending_calls
            ],
            return_exceptions=True,
        )

        merged_results: list[dict[str, str]] = []
        for (call_name, _), call_result in zip(
            pending_calls,
            call_results,
        ):
            if isinstance(call_result, Exception):
                merged_results.append(
                    {
                        "name": call_name,
                        "status": "failed",
                        "error": str(call_result),
                    }
                )
                continue

            merged_results.append(
                {
                    "name": call_name,
                    "status": "completed",
                    "result": _extract_tool_response_text(call_result),
                }
            )

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=json.dumps(
                        {"results": merged_results},
                        ensure_ascii=False,
                    ),
                ),
            ],
        )

    toolkit.register_tool_function(batch_run_tool)

    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(subscribe_system_time)
    toolkit.register_tool_function(wait_task_result)

    agent = ReActAgent(
        name="Friday",
        sys_prompt="你是一个名叫Friday的中文对话个人助手。",
        model=OpenAIChatModel(
            model_name="stepfun/step-3.5-flash:free",
            api_key=os.environ["OPENROUTER_API_KEY"],
            client_kwargs={"base_url": "https://openrouter.ai/api/v1"},
            stream=True,
        ),
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
    )

    msg = Msg(
        name="user",
        content=(
            "你好，Friday！请按以下步骤操作："
            "1）使用batch_run_tool，同时调用三个工具："
            "第一个execute_python_code先sleep 5秒后输出系统时间；"
            "第二个execute_python_code先sleep 10秒后输出Python系统版本；"
            "第三个subscribe_system_time每隔4秒订阅一次系统时间；"
            "两个execute_python_code都要设置blocking=False并且subscribe=True；"
            "2）然后调用wait_task_result，不传task_ids，等待并获取第一组订阅结果；"
            "3）再调用一次wait_task_result，不传task_ids，获取第二组订阅结果；"
            "4）第一个代码必须包含time.sleep(5)，第二个代码必须包含time.sleep(10)；"
            "5）最后用中文分别告诉我两个结果。"
        ),
        role="user",
    )
    
    while True:
        msg = await agent(msg)
        if msg.get_text_content() == "exit":
            break
        break

if __name__ == "__main__":
    """
    将异步启动参数从run_tool移除，现在只由工具函数本身决定是否异步执行，
    并且添加了一个新的工具函数subscribe_system_time测试工具，
    用来测试wait_task_result批量获取订阅结果的能力
    演示了在非阻塞模式下工具函数主动推送结果的能力。
    """
    asyncio.run(main())




