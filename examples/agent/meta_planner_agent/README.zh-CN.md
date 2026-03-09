# 元规划代理示例

在此示例中，我们演示了

- 如何构建一个规划代理，将复杂任务分解为可管理的子任务，并协调子代理完成这些任务
- 如何在多代理系统中正确处理子代理的打印消息
- 如何将子代理的中断事件传递给主规划代理

具体来说，在 [main.py](./main.py) 中，创建了一个带有 `PlanNotebook` 实例的规划代理，用于创建和管理计划。它配备了一个名为 `create_worker` 的工具函数（见 [tool.py](./tool.py)），可动态创建子代理并完成分配的子任务。这些子代理配备了一些基本工具，以及一些预设的 MCP 服务器，以增强它们的能力。

> 我们建议使用 AgentScope-Studio 来可视化本示例中的代理交互。

## 快速开始

如果尚未安装 agentscope，请安装：

```bash
pip install agentscope
```

确保你已设置好 DashScope将 API 密钥作为环境变量。

在此示例中，子代理配备了以下 MCP 服务器，设置相应的环境变量以激活它们。
如果未设置，相应的 MCP 将被禁用。
有关这些工具的更多详细信息，请参考 [tool.py](./tool.py)。您也可以根据需要添加或修改工具。

| MCP                       | 描述                               | 环境变量             |
|---------------------------|-----------------------------------|--------------------|
| 高德 MCP                  | 提供地图相关服务                   | GAODE_API_KEY       |
| GitHub MCP                | 搜索和访问 GitHub 仓库             | GITHUB_TOKEN       |
| Microsoft Playwright MCP  | 用于网页浏览的 MCP 服务器          | -                  |

运行示例：

```bash
python main.py
```

然后您可以让规划代理帮助您完成复杂任务，例如“对 AgentScope 仓库进行研究”。

简单问题的注意事项，规划代理可以直接回答，而无需创建子代理。

## 高级用法

### 处理子代理输出

在这个示例中，子代理不会直接在控制台打印信息（通过在 tool.py 中的 `agent.set_console_output_enable(True)`）。
相反，其打印信息会被简化并返回给规划代理，作为工具函数 `create_worker` 的流式响应。
这样，我们只向用户展示规划代理，而不是多个代理，从而提供更好的用户体验。
然而，如果子代理在完成分配任务时进行了长时间的推理-操作过程，工具函数 `create_worker` 的响应可能会占用过多的上下文长度。

下图展示了子代理输出如何在 AgentScope-Studio 中作为工具流式响应显示：

<details>
 <summary>中文</summary>
 <p align="center">
  <img src="./assets/screenshot_zh.jpg"/>
 </p>
</details>

<details>
 <summary>英文</summary>
 <p align="center">
  <imgsrc="./assets/screenshot_en.jpg"/>
 </p>
</details>

另外，你可以选择将子代理展示给用户，并且只将结构化结果作为 `create_worker` 的工具结果返回给规划代理。

### 传播中断事件

在 `ReActAgent` 中，当最终答案通过 `handle_interrupt` 函数生成时，返回消息的元数据字段将包含一个 `_is_interrupted` 键，其值为 `True`，以表示代理已被中断。

通过这个字段，我们可以在工具函数 `create_worker` 中将中断事件从子代理传播到主规划代理。对于用户自定义的代理类，你可以在代理类的 `handle_interrupt` 函数中定义你自己的传播逻辑。

### 更改 LLM

这个示例是使用 DashScope 聊天模型构建的。如果你想在此示例中更换模型，不要忘记同时更改格式化器！内置模型之间的对应关系为格式化程序在[我们的教程](https://doc.agentscope.io/tutorial/task_prompt.html#id1)中列出

## 进一步阅读

- [计划](https://doc.agentscope.io/tutorial/task_plan.html)