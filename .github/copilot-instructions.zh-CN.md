# AgentScope 代码审查指南

您应该进行严格的代码审查。每条要求都标有优先级：
- **[MUST]** 必须满足，否则 PR 将被拒绝
- **[SHOULD]** 强烈建议
- **[MAY]** 可选建议

## 1. 代码质量

### [MUST] 延迟加载
- 第三方库依赖应在使用时导入，避免在文件顶部集中导入
  - “第三方库”指的是未包含在 `pyproject.toml` 文件的 `dependencies` 变量中的库。
- 对于基类导入，使用工厂模式：
```python
def get_xxx_cls() -> "MyClass":
    from xxx import BaseClass
    class MyClass(BaseClass): ...
    return MyClass
```

### [SHOULD] 代码简洁性
在理解代码意图后，检查是否可以优化：
- 避免不必要的临时变量
- 合并重复代码块
- 优先复用已有的工具函数

### [MUST] 封装标准
- 所有 Python 文件下`src/agentscope` 应该以 `_` 前缀命名，并通过 `__init__.py` 控制暴露
- 框架内部使用且不需要暴露给用户的类和函数必须以 `_` 前缀命名

## 2. [必须] 代码安全
- 禁止硬编码 API 密钥/令牌/密码
- 使用环境变量或配置文件进行管理
- 检查调试信息和临时凭证
- 检查注入攻击风险（SQL/命令/代码注入等）

## 3. [必须] 测试与依赖
- 新功能必须包含单元测试
- 新依赖需要添加到 `pyproject.toml` 对应部分
- 非核心场景的依赖不应添加到最小依赖列表中

## 4. 代码规范

### [必须] 注释规范
- **使用英语**
- 所有类/方法必须有完整的文档字符串，严格遵循模板：
```python
def func(a: str, b: int | None = None) -> str:
    """{description}"""参数：
        a (`str`)：
            参数 a
        b (`int | None`，可选)：
            参数 b

    返回：
        `str`：
            返回的字符串
    """
```
- 对特殊内容使用 reStructuredText 语法：
```python
class MyClass:
    """xxx

    `示例链接 <https://xxx>`_

    .. note:: 示例注释

    .. tip:: 示例小贴士

    .. important:: 示例重要信息

    .. code-block:: python

        def hello_world():
            print("Hello world!")

    """
```

### [必须] 提交前检查
- **严格审查**：在大多数情况下，应修改代码而不是跳过检查
- 禁止跳过文件级检查
- 唯一允许的跳过：代理类系统提示参数（以避免 `\n` 格式问题）

---

## 5. Git 标准

### [必须] PR 标题
- 遵循 Conventional Commits 标准
- 必须使用前缀：`feat/fix/docs/ci/refactor/test` 等
- 格式：`feat(scope): description`
- 示例：`feat(memory): 添加 redis 缓存支持`