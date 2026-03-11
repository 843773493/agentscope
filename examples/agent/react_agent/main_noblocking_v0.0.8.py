# -*- coding: utf-8 -*-
"""The main entry point of the ReAct agent example."""
import asyncio
import datetime
import inspect
import json
import os
import sys
import tempfile
from functools import partial
from typing import Any

import shortuuid
from dotenv import load_dotenv

from agentscope.agent import CustomReActAgent
# CompressionConfig is defined in the internal react agent module
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


class AgentRuntimeContext:
    """Store per-agent background-task and subscription state."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._subscription_result_queues: dict[
            str,
            asyncio.Queue[dict[str, Any]],
        ] = {
            "default": asyncio.Queue(),
            "notify": asyncio.Queue(),
        }

    def create_task_record(
        self,
        *,
        task_id: str,
        created_at: str,
        message: str,
        subscribe: bool,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Create, store, and return metadata for one background task."""

        metadata: dict[str, Any] = {
            "status": "queued",
            "created_at": created_at,
            "start_iso_time": created_at,
            "last_received_iso_time": None,
            "message": message,
            "future": None,
            "subscribe": subscribe,
            **extra_fields,
        }
        self._tasks[task_id] = metadata
        return metadata

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Return stored metadata for one task id."""

        return self._tasks.get(task_id)

    def set_task_future(
        self,
        task_id: str,
        future: asyncio.Task[Any],
    ) -> None:
        """Attach the created asyncio task to its metadata."""

        self._tasks[task_id]["future"] = future

    def select_subscription_queue_name(
        self,
        subscription_message: dict[str, Any],
    ) -> str:
        """Select the target subscription queue name."""

        if subscription_message.get("notify") is True:
            return "notify"
        return "default"

    async def publish_subscription_message(
        self,
        subscription_message: dict[str, Any],
    ) -> None:
        """Publish one subscription message into the selected queue."""

        queue_name = self.select_subscription_queue_name(
            subscription_message,
        )
        await self._subscription_result_queues[queue_name].put(
            subscription_message,
        )

    async def get_subscription_message(
        self,
        include_notify: bool = True,
    ) -> dict[str, Any]:
        """Wait until one subscription message is available."""

        get_tasks = [
            asyncio.create_task(
                self._subscription_result_queues["default"].get(),
            ),
        ]
        if include_notify:
            get_tasks.append(
                asyncio.create_task(
                    self._subscription_result_queues["notify"].get(),
                )
            )

        done, pending = await asyncio.wait(
            get_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for pending_task in pending:
            pending_task.cancel()

        completed_task = next(iter(done))
        return completed_task.result()

    def pop_subscription_message_nowait(
        self,
        queue_name: str,
    ) -> dict[str, Any]:
        """Return one queued subscription message without waiting."""

        return self._subscription_result_queues[queue_name].get_nowait()

    async def put_subscription_message(
        self,
        queue_name: str,
        one_message: dict[str, Any],
    ) -> None:
        """Put one message back into a subscription queue."""

        await self._subscription_result_queues[queue_name].put(one_message)

    def touch_message_timestamp(
        self,
        one_message: dict[str, Any],
    ) -> None:
        """Refresh one subscription payload with the latest receive time."""

        start_iso_time = one_message.get("start_iso_time")
        last_received_iso_time = _now_iso()

        one_message["start_iso_time"] = start_iso_time
        one_message["last_received_iso_time"] = last_received_iso_time

        task_id = one_message.get("task_id")
        if isinstance(task_id, str) and task_id in self._tasks:
            self._tasks[task_id]["last_received_iso_time"] = (
                last_received_iso_time
            )


class GroupChatService:
    """Manage shared chat rooms and fan-out messages across agents."""

    def __init__(self) -> None:
        self._room_subscribers: dict[
            str,
            set[asyncio.Queue[dict[str, Any]]],
        ] = {}
        self.add_room("global")

    def add_room(self, room_name: str) -> None:
        """Add a room if it does not exist."""

        if room_name not in self._room_subscribers:
            self._room_subscribers[room_name] = set()

    def subscribe(
        self,
        room_name: str,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Create one subscriber queue for a room."""

        self.add_room(room_name)
        subscriber_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._room_subscribers[room_name].add(subscriber_queue)
        return subscriber_queue

    def unsubscribe(
        self,
        room_name: str,
        subscriber_queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        """Remove one subscriber queue from a room."""

        subscribers = self._room_subscribers.get(room_name)
        if subscribers is None:
            return

        subscribers.discard(subscriber_queue)

    async def send_message(
        self,
        sender_name: str,
        room_name: str,
        text_message: str,
    ) -> tuple[bool, int]:
        """Publish a room message to all subscribers in that room."""

        subscribers = self._room_subscribers.get(room_name)
        if subscribers is None:
            return False, 0

        payload = {
            "sender_name": sender_name,
            "room_name": room_name,
            "text_message": text_message,
            "sent_iso_time": _now_iso(),
        }

        delivered_count = 0
        for subscriber_queue in list(subscribers):
            await subscriber_queue.put(payload)
            delivered_count += 1

        return True, delivered_count

async def execute_python_code(
    code: str,
    timeout: float = 300,
    blocking: bool = True,
    subscribe: bool = False,
    runtime_context: AgentRuntimeContext | None = None,
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

    if runtime_context is None:
        raise ValueError("runtime_context is required.")

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
    metadata = runtime_context.create_task_record(
        task_id=task_id,
        created_at=created_at,
        message="Task started successfully.",
        subscribe=parsed_subscribe,
    )

    async def _wrapper() -> ToolResponse:
        metadata["status"] = "running"
        try:
            result = await _execute_once()
            metadata["status"] = "completed"
            metadata["result"] = result

            if metadata["subscribe"]:
                last_received_iso_time = _now_iso()
                metadata["last_received_iso_time"] = last_received_iso_time
                await runtime_context.publish_subscription_message(
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
                await runtime_context.publish_subscription_message(
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
    runtime_context.set_task_future(task_id, fut)

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
    notify_reminder: bool = False,
    runtime_context: AgentRuntimeContext | None = None,
    **kwargs: Any,
) -> ToolResponse:
    """Subscribe system time updates and publish them to the queue.

    Args:
        interval_seconds (`float`, defaults to `4`):
            Interval in seconds between two system time updates.
        notify_reminder (`bool`, defaults to `False`):
            Whether to emit reminder-style notifications that can be consumed
            by ``wait_task_result`` when waiting by ``task_ids``.

    Returns:
        `ToolResponse`:
            Task metadata for the background subscription publisher.
    """

    if runtime_context is None:
        raise ValueError("runtime_context is required.")

    parsed_interval = _parse_json_if_needed(interval_seconds)
    parsed_notify_reminder = _parse_json_if_needed(notify_reminder)

    if not isinstance(parsed_interval, (int, float)):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid interval_seconds format: expected number.",
                ),
            ],
        )

    if not isinstance(parsed_notify_reminder, bool):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid notify_reminder format: expected bool.",
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
    metadata = runtime_context.create_task_record(
        task_id=task_id,
        created_at=created_at,
        message="System time subscription started.",
        subscribe=True,
    )

    async def _publisher() -> None:
        metadata["status"] = "running"
        try:
            while True:
                await asyncio.sleep(interval)
                last_received_iso_time = _now_iso()
                metadata["last_received_iso_time"] = last_received_iso_time

                await runtime_context.publish_subscription_message(
                    {
                        "task_id": task_id,
                        "name": "subscribe_system_time",
                        "status": "streaming",
                        "result": last_received_iso_time,
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": last_received_iso_time,
                        "blocking": False,
                        "notify": parsed_notify_reminder,
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
            await runtime_context.publish_subscription_message(
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
    runtime_context.set_task_future(task_id, fut)

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
                        "notify": parsed_notify_reminder,
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )

async def subscribe_group_chat_messages(
    room_name: str = "global",
    notify_reminder: bool = False,
    runtime_context: AgentRuntimeContext | None = None,
    group_chat_service: GroupChatService | None = None,
    **kwargs: Any,
) -> ToolResponse:
    """Subscribe group chat messages and publish them to wait queues.

    Args:
        room_name (`str`, defaults to `"global"`):
            The target chat room name.
        notify_reminder (`bool`, defaults to `False`):
            Whether to route updates to the reminder queue.

    Returns:
        `ToolResponse`:
            Task metadata for the background room subscription.
    """

    if runtime_context is None:
        raise ValueError("runtime_context is required.")
    if group_chat_service is None:
        raise ValueError("group_chat_service is required.")

    parsed_room_name = _parse_json_if_needed(room_name)
    parsed_notify_reminder = _parse_json_if_needed(notify_reminder)

    if not isinstance(parsed_room_name, str) or not parsed_room_name.strip():
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid room_name format: expected non-empty str.",
                ),
            ],
        )

    if not isinstance(parsed_notify_reminder, bool):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid notify_reminder format: expected bool.",
                ),
            ],
        )

    normalized_room_name = parsed_room_name.strip()
    group_chat_service.add_room(normalized_room_name)

    task_id = shortuuid.uuid()
    created_at = _now_iso()
    metadata = runtime_context.create_task_record(
        task_id=task_id,
        created_at=created_at,
        message="Group chat subscription started.",
        subscribe=True,
        room_name=normalized_room_name,
    )

    subscriber_queue = group_chat_service.subscribe(normalized_room_name)

    async def _publisher() -> None:
        metadata["status"] = "running"
        try:
            while True:
                one_message = await subscriber_queue.get()
                last_received_iso_time = _now_iso()
                metadata["last_received_iso_time"] = last_received_iso_time

                await runtime_context.publish_subscription_message(
                    {
                        "task_id": task_id,
                        "name": "subscribe_group_chat_messages",
                        "status": "message_received",
                        "room_name": normalized_room_name,
                        "message": one_message,
                        "result": one_message.get("text_message"),
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": last_received_iso_time,
                        "blocking": True,
                        "notify": parsed_notify_reminder,
                    }
                )
        except asyncio.CancelledError:  # pragma: no cover - example code
            metadata["status"] = "cancelled"
            metadata["message"] = "Group chat subscription cancelled."
            raise
        except Exception as exc:  # pragma: no cover - example code
            metadata["status"] = "failed"
            metadata["message"] = str(exc)
            last_received_iso_time = _now_iso()
            metadata["last_received_iso_time"] = last_received_iso_time
            await runtime_context.publish_subscription_message(
                {
                    "task_id": task_id,
                    "name": "subscribe_group_chat_messages",
                    "status": "failed",
                    "error": str(exc),
                    "room_name": normalized_room_name,
                    "start_iso_time": metadata["start_iso_time"],
                    "last_received_iso_time": last_received_iso_time,
                    "notify": parsed_notify_reminder,
                }
            )
            raise
        finally:
            group_chat_service.unsubscribe(
                normalized_room_name,
                subscriber_queue,
            )

    fut = asyncio.create_task(_publisher())
    runtime_context.set_task_future(task_id, fut)

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "task_id": task_id,
                        "name": "subscribe_group_chat_messages",
                        "status": metadata["status"],
                        "created_at": created_at,
                        "message": metadata["message"],
                        "room_name": normalized_room_name,
                        "start_iso_time": metadata["start_iso_time"],
                        "last_received_iso_time": metadata[
                            "last_received_iso_time"
                        ],
                        "blocking": False,
                        "notify": parsed_notify_reminder,
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )

async def send_group_chat_message(
    room_name: str = "global",
    text_message: str = "",
    agent: CustomReActAgent | None = None,
    group_chat_service: GroupChatService | None = None,
    **kwargs: Any,
) -> ToolResponse:
    """Send one text message to a target group chat room.

    Args:
        room_name (`str`, defaults to `"global"`):
            The target room name.
        text_message (`str`, defaults to `""`):
            The message content.

    Returns:
        `ToolResponse`:
            A JSON object that indicates whether the send succeeds.
    """
    if agent is None:
        raise ValueError("agent is required.")
    if group_chat_service is None:
        raise ValueError("group_chat_service is required.")

    sender_name = agent.name

    parsed_sender_name = _parse_json_if_needed(sender_name)
    parsed_room_name = _parse_json_if_needed(room_name)
    parsed_text_message = _parse_json_if_needed(text_message)

    if not isinstance(parsed_sender_name, str) or not parsed_sender_name.strip():
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid sender_name format: expected non-empty str.",
                ),
            ],
        )

    if not isinstance(parsed_room_name, str) or not parsed_room_name.strip():
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid room_name format: expected non-empty str.",
                ),
            ],
        )

    if not isinstance(parsed_text_message, str) or not parsed_text_message.strip():
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid text_message format: expected non-empty str.",
                ),
            ],
        )

    normalized_sender_name = parsed_sender_name.strip()
    normalized_room_name = parsed_room_name.strip()
    normalized_text_message = parsed_text_message.strip()

    is_success, delivered_count = await group_chat_service.send_message(
        sender_name=normalized_sender_name,
        room_name=normalized_room_name,
        text_message=normalized_text_message,
    )

    status_message = "Message sent successfully."
    if not is_success:
        status_message = (
            f"Message send failed: room {normalized_room_name} not found."
        )

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "success": is_success,
                        "sender_name": normalized_sender_name,
                        "room_name": normalized_room_name,
                        "text_message": normalized_text_message,
                        "delivered_subscribers": delivered_count,
                        "message": status_message,
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )

async def batch_run_tool(
    calls: list[dict[str, Any]] | str,
    toolkit: Toolkit,
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

        # batch_run_tool previously expected parameters under the key
        # "input". some clients (or older versions) may send
        # "arguments" instead, so allow either and normalise to input.
        call_input = None
        if "input" in call_item:
            call_input = call_item.get("input")
        elif "arguments" in call_item:
            call_input = call_item.get("arguments")

        pending_calls.append((call_name, call_input))

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

async def wait_task_result(
    task_ids: list[str],
    runtime_context: AgentRuntimeContext | None = None,
) -> ToolResponse:
    """Wait for background tasks to complete.

    Args:
        task_ids (`list[str]`):
            Task ids to wait for. If empty, this call waits for one
            subscribed result and returns it.

    Returns:
        `ToolResponse`:
            A JSON summary of waited task results, or one subscribed result.

    Notes:
        The following collected fields affect result gathering behavior:

        task_id (`str`):
            In ``task_ids`` mode, waited results are keyed by task id. For
            subscription queues, ``notify`` queue messages are always
            collected, while ``default`` queue messages are only collected
            when their ``task_id`` is in target ``task_ids``.

        subscribe (`bool`):
            If upstream tools set ``subscribe=True``, completion or failure
            events are published to subscription queues. If ``False``, no
            subscription event is emitted and this function mainly depends on
            awaited futures in ``task_ids`` mode.

        blocking (`bool`):
            Used in subscription-message mode (empty ``task_ids``). Messages
            with ``blocking=False`` are treated as streaming updates and this
            function keeps collecting. It returns when a message has
            ``blocking`` not equal to ``False``.

        notify (`bool`):
            Controls queue routing before collection. ``notify=True`` routes
            messages to the ``notify`` queue and they are always consumed in
            ``task_ids`` mode; otherwise messages go to ``default`` queue and
            are filtered by target ``task_id`` rules.
    """

    if runtime_context is None:
        raise ValueError("runtime_context is required.")

    normalized_ids: list[str] = []

    def _append_subscription_payload(
        payload: list[dict[str, Any]],
        one_message: dict[str, Any],
    ) -> None:
        payload.append(
            {
                "task_id": one_message.get("task_id"),
                "status": "subscribed",
                "subscription": one_message,
                "start_iso_time": one_message.get("start_iso_time"),
                "last_received_iso_time": one_message.get(
                    "last_received_iso_time"
                ),
            }
        )

    async def _collect_subscription_messages_for_task_ids(
        target_task_ids: set[str],
    ) -> list[dict[str, Any]]:
        matched_messages: list[dict[str, Any]] = []

        # notify queue is always collected for task_ids mode.
        while True:
            try:
                notify_message = runtime_context.pop_subscription_message_nowait(
                    "notify",
                )
            except asyncio.QueueEmpty:
                break

            runtime_context.touch_message_timestamp(notify_message)
            matched_messages.append(notify_message)

        # default queue is only collected for matched target task ids and
        # excludes execute_python_code terminal statuses to avoid duplicates
        # with waited future results.
        unmatched_default_messages: list[dict[str, Any]] = []
        while True:
            try:
                one_message = runtime_context.pop_subscription_message_nowait(
                    "default",
                )
            except asyncio.QueueEmpty:
                break

            task_id = one_message.get("task_id")
            is_target_task_message = (
                isinstance(task_id, str) and task_id in target_task_ids
            )
            is_python_terminal_message = (
                one_message.get("name") == "execute_python_code"
                and one_message.get("status") in {"completed", "failed"}
            )

            if is_target_task_message and not is_python_terminal_message:
                runtime_context.touch_message_timestamp(one_message)
                matched_messages.append(one_message)
            else:
                unmatched_default_messages.append(one_message)

        for one_message in unmatched_default_messages:
            await runtime_context.put_subscription_message(
                "default",
                one_message,
            )

        return matched_messages

    async def _wait_subscription_results() -> ToolResponse:
        buffered_results: list[dict[str, Any]] = []

        while True:
            subscription_result = await runtime_context.get_subscription_message(
                include_notify=True,
            )
            runtime_context.touch_message_timestamp(subscription_result)

            buffered_results.append(subscription_result)

            # 除非消息中blocking参数为True，否则继续收集
            if (
                subscription_result.get("blocking") is False
            ):
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

    if not isinstance(task_ids, list):
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Invalid task_ids format: expected list[str].",
                ),
            ],
        )

    # Subscription mode: when task ids are empty, block until any subscribed
    # result arrives, then return exactly one result.
    if not task_ids:
        return await _wait_subscription_results()

    # 校验 task_ids 中每一项必须为字符串。
    for index, item in enumerate(task_ids):
        if not isinstance(item, str):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            "Invalid task_ids item at index "
                            f"{index}: expected str."
                        ),
                    ),
                ],
            )
        normalized_ids.append(item)

    # 去重并保持输入顺序，避免重复等待同一任务。
    deduped_ids: list[str] = []
    seen: set[str] = set()
    for item in normalized_ids:
        if item and item not in seen:
            seen.add(item)
            deduped_ids.append(item)

    # 去重后为空，退化为主动等待订阅结果模式。
    if not deduped_ids:
        return await _wait_subscription_results()

    async def _wait_one(
        one_task_id: str,
    ) -> dict[str, Any]:
        info = runtime_context.get_task(one_task_id)
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

    # 并发等待所有目标 task_ids 的结果。
    waited_results = await asyncio.gather(
        *[_wait_one(one_task_id) for one_task_id in deduped_ids],
    )

    # 在任务结果就绪后，补收当前已到达的订阅提醒消息。
    matched_subscription_messages = (
        await _collect_subscription_messages_for_task_ids(
            set(deduped_ids),
        )
    )
    subscription_payload: list[dict[str, Any]] = []
    for one_message in matched_subscription_messages:
        _append_subscription_payload(subscription_payload, one_message)

    # 先写入各 task 的执行结果。
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

    # 再追加订阅提醒结果，统一一次返回给调用方。
    payload.extend(subscription_payload)

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps({"results": payload}, ensure_ascii=False),
            ),
        ],
    )


def _build_agent(
    name: str,
    sys_prompt: str,
    runtime_context: AgentRuntimeContext,
    group_chat_service: GroupChatService,
) -> CustomReActAgent:
    """Create one demo agent and pre-bind tool dependencies."""

    agent = CustomReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=OpenAIChatModel(
            model_name="stepfun/step-3.5-flash:free",
            api_key=os.environ["OPENROUTER_API_KEY"],
            client_kwargs={"base_url": "https://openrouter.ai/api/v1"},
            stream=True,
        ),
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
        toolkit=Toolkit(),
    )
    agent.runtime_context = runtime_context
    agent.group_chat_service = group_chat_service

    agent.toolkit.register_tool_function(
        partial(batch_run_tool, toolkit=agent.toolkit),
    )
    agent.toolkit.register_tool_function(
        partial(execute_python_code, runtime_context=agent.runtime_context),
    )
    agent.toolkit.register_tool_function(
        partial(subscribe_system_time, runtime_context=agent.runtime_context),
    )
    agent.toolkit.register_tool_function(
        partial(
            subscribe_group_chat_messages,
            runtime_context=agent.runtime_context,
            group_chat_service=agent.group_chat_service,
        ),
    )
    agent.toolkit.register_tool_function(
        partial(
            send_group_chat_message,
            agent=agent,
            group_chat_service=agent.group_chat_service,
        ),
    )
    agent.toolkit.register_tool_function(
        partial(wait_task_result, runtime_context=agent.runtime_context),
    )
    return agent

async def main() -> None:
    """The main entry point for the ReAct agent example."""
    load_dotenv()

    group_chat_service = GroupChatService()
    friday_agent = _build_agent(
        name="Friday",
        sys_prompt="你是一个名叫Friday的中文对话个人助手。",
        runtime_context=AgentRuntimeContext(),
        group_chat_service=group_chat_service,
    )

    msg = Msg(
        name="user",
        content=(
            "你好，Friday！请按以下步骤完成群聊工具测试："
            "1）先通过batch_run_tool调用subscribe_group_chat_messages，"
            "传入room_name='global'完成订阅；"
            "2）订阅成功后，再调用send_group_chat_message，"
            "传入sender_name='Friday'、room_name='global'、"
            "text_message='这是一条群聊测试消息'；"
            "3）然后调用wait_task_result，主动传入task_ids=[]，"
            "接收一条刚刚订阅到的群聊消息；"
            "4）最后用中文反馈：发送是否成功、接收到的消息内容、"
            "以及接收消息是否与发送内容一致。"
        ),
        role="user",
    )
    
    _sunday_agent = _build_agent(
        name="Sunday",
        sys_prompt="你是一个名叫Sunday的中文对话个人助手。",
        runtime_context=AgentRuntimeContext(),
        group_chat_service=group_chat_service,
    )
    
    _msg_2 = Msg(
        name="user",
        content=(
            "你好，Sunday！请按以下步骤完成群聊工具测试："
            "1）先调用subscribe_group_chat_messages，"
            "传入room_name='global'完成订阅；"
            "2）订阅成功后，再调用send_group_chat_message，"
            "传入sender_name='Sunday'、room_name='global'、"
            "text_message='这是一条群聊测试消息'；"
            "3）然后调用wait_task_result，主动传入task_ids=[]，"
            "接收一条刚刚订阅到的群聊消息；"
            "4）最后用中文反馈：发送是否成功、接收到的消息内容、"
            "以及接收消息是否与发送内容一致。"
        ),
        role="user",
    )
    
    while True:
        await asyncio.gather(
            friday_agent(msg),
            # agent_2(msg_2),
        )
        break

if __name__ == "__main__":
    """
    改为使用自定义的agent，默认将模型输出打印在data\temp\Friday.txt
    改变工具解析方法，支持partial方式传入工具参数为对象，提高工具访问的灵活性和可扩展性
    将_task等运行时参数整合到AgentRuntimeContext中，将chatroom封装为GroupChatService，
    最后作为CustomReActAgent内部属性便于自定义工具函数访问任意数据
    """
    asyncio.run(main())



