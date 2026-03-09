# -*- coding: utf-8 -*-
"""The main entry point of the ReAct agent example."""
import asyncio
import datetime
import inspect
import json
import os
import sys
import tempfile
from typing import Any, Callable

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


# placeholder; actual run_tool defined inside main after toolkit creation

async def execute_python_code(
    code: str,
    timeout: float = 300,
    **kwargs: Any,
) -> ToolResponse:
    """Execute the given python code in a temp file and capture the return
    code, standard output and error. Note you must `print` the output to get
    the result, and the tmp file will be removed right after the execution.

    Args:
        code (`str`):
            The Python code to be executed.
        timeout (`float`, defaults to `300`):
            The maximum time (in seconds) allowed for the code to run.

    Returns:
        `ToolResponse`:
            The response containing the return code, standard output, and
            standard error of the executed code.
    """

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

    # ``run_tool`` is a thin wrapper around the toolkit itself.  rather than
    # accepting a callable, it takes the *parameters* of a tool invocation
    # (name + input) and decides whether to wait for completion depending on
    # the ``blocking`` flag.
    async def run_tool(
        blocking: bool,
        name: str,
        input: dict | None = None,
    ) -> ToolResponse:
        """Invoke another registered tool, optionally in the background.

        Args:
            blocking (`bool`): if ``True`` wait for the tool to finish before
                returning.  If ``False`` the request is scheduled and a
                metadata document (containing ``task_id`` etc.) is returned
                immediately.
            name (`str`): the name of the tool function to call.
            input (`dict | None`, optional): the arguments to supply to the
                tool, equivalent to the ``input`` field of a ``ToolUseBlock``.
        """

        # helper that actually executes the tool and accumulates the final
        # chunk as a single ``ToolResponse`` so callers get a normal object
        async def _execute_once() -> ToolResponse:
            # The agent sometimes serializes the input dict as a JSON string,
            # so try to decode it back to a mapping for convenience.
            parsed_input = input
            if isinstance(parsed_input, str):
                try:
                    parsed_input = json.loads(parsed_input)
                except json.JSONDecodeError:
                    pass

            call = {"name": name, "input": parsed_input}
            final: ToolResponse | None = None

            # ``call_tool_function`` is supposed to return an async generator,
            # but due to decorators it may occasionally masquerade as a
            # coroutine.  handle both cases gracefully to avoid warnings/errors.
            res = toolkit.call_tool_function(call)
            if inspect.isasyncgen(res):
                async for chunk in res:  # type: ignore[arg-type]
                    final = chunk
            elif inspect.isawaitable(res):
                # e.g. tracing disabled returned a coroutine
                final = await res  # type: ignore[assignment]
            else:
                # unexpected type, just treat it as direct result
                final = res  # type: ignore[assignment]

            return final or ToolResponse(content=[])

        if blocking:
            return await _execute_once()

        # non-blocking path: mirror the pattern used for execute_python_code
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
                result = await _execute_once()
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

    toolkit.register_tool_function(run_tool)
    
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
        break

if __name__ == "__main__":
    """
    在异步工具的基础上添加，添加一个包装其它工具为异步执行的包装工具run_tool
    """
    asyncio.run(main())


