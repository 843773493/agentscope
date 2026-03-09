# -*- coding: utf-8 -*-
"""The main entry point of the ReAct agent example."""
import asyncio
import datetime
import json
import os
import sys
import tempfile
from typing import Any

import shortuuid
from dotenv import load_dotenv

from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import DashScopeChatFormatter, OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import DashScopeChatModel, OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

# simple in-memory structure for tracking spawned tasks when blocking=False
# each entry maps task_id -> metadata dict containing status, timestamps and
# the underlying asyncio.Task object.  This is intentionally minimal; in a
# production environment you'd want persistence, locks, cancellation support,
# etc.
_tasks: dict[str, dict] = {}


async def _run_python_code_inner(code: str, timeout: float) -> ToolResponse:
    """Actual executor used by both blocking and non-blocking paths."""

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


async def execute_python_code(
    code: str,
    timeout: float = 300,
    blocking: bool = True,
    **kwargs: Any,
) -> ToolResponse:
    """Run python code either synchronously or as a background task.

    * When ``blocking`` is ``True`` the helper ``_run_python_code_inner`` is
      invoked directly and the caller receives the ``ToolResponse`` produced by
      the execution.
    * When ``blocking`` is ``False`` the work is scheduled via
      ``asyncio.create_task`` and a small metadata document is returned
      immediately.  The caller can subsequently call ``wait_task_result`` with
      the returned ``task_id`` to receive the actual execution result.

    The metadata object contains::

        {
            "task_id": "<unique>",
            "status": "queued",        # quickly becomes "running"
            "created_at": "<iso timestamp>",
            "message": "Task started successfully."
        }

    Args:
        code (`str`): code to execute.
        timeout (`float`, optional): max seconds allowed (default 300).
        blocking (`bool`, optional): run synchronously if True.

    Returns:
        `ToolResponse`: either the execution result or the task metadata.
    """

    if blocking:
        return await _run_python_code_inner(code, timeout)

    # --- non-blocking path -------------------------------------------------
    task_id = shortuuid.uuid()
    created_at = datetime.datetime.now().isoformat()
    metadata: dict = {
        "status": "queued",
        "created_at": created_at,
        "message": "Task started successfully.",
        "future": None,
    }

    async def _wrapper() -> ToolResponse:
        metadata["status"] = "running"
        try:
            result = await _run_python_code_inner(code, timeout)
            metadata["status"] = "completed"
            metadata["result"] = result
            return result
        except Exception as exc:  # pragma: no cover - example code
            metadata["status"] = "failed"
            metadata["message"] = str(exc)
            raise

    fut = asyncio.create_task(_wrapper())
    metadata["future"] = fut
    _tasks[task_id] = metadata

    # return metadata as a simple JSON string for readability in examples
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "task_id": task_id,
                        "status": metadata["status"],
                        "created_at": created_at,
                        "message": metadata["message"],
                    }
                ),
            ),
        ],
    )


async def wait_task_result(task_id: str) -> ToolResponse:
    """Wait until the background task with ``task_id`` completes.

    Returns the same ``ToolResponse`` that the original execution produced.
    """

    info = _tasks.get(task_id)
    if info is None:
        return ToolResponse(content=[TextBlock(type="text", text=f"Task {task_id} not found.")])

    fut = info.get("future")
    if fut is None:
        return ToolResponse(content=[TextBlock(type="text", text="No associated future for task.")])

    try:
        result = await fut
    except Exception as exc:
        return ToolResponse(content=[TextBlock(type="text", text=f"Task failed: {exc}")])

    return result


async def main() -> None:
    """The main entry point for the ReAct agent example."""
    load_dotenv()
    toolkit = Toolkit()

    toolkit.register_tool_function(execute_python_code)
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
        content="你好，Friday！请使用非blocking方式执行python代码查看系统时间，在代码执行完成后，使用工具函数wait_task_result查询结果并告诉我。",
        role="user",
    )
    while True:
        msg = await agent(msg)
        if msg.get_text_content() == "exit":
            break

if __name__ == "__main__":
    """
    想法的初次实现，给工具添加异步参数，并通过另一个工具收集异步结果
    """
    asyncio.run(main())


