# 支持新模型

目前我们已经支持的模型有 HF 模型、部分模型 API 、部分第三方模型。

## 新增API模型

新增基于API的模型，需要在 `opencompass/models` 下新建 `mymodel_api.py` 文件，继承 `BaseAPIModel`，并实现 `generate` 方法来进行推理，以及 `get_token_len` 方法来计算 token 的长度。在定义好之后修改对应配置文件名称即可。

```python
from ..base_api import BaseAPIModel

class MyModelAPI(BaseAPIModel):

    is_api: bool = True

    def __init__(self,
                 path: str,
                 max_seq_len: int = 2048,
                 query_per_second: int = 1,
                 retry: int = 2,
                 **kwargs):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         retry=retry)
        ...

    def generate(
        self,
        inputs,
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        pass

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string."""
        pass
```

### OpenAI 兼容 API 的快速接入示例

如果你的服务兼容 OpenAI Chat Completions 接口，通常无需新增 `opencompass/models/*.py`，直接使用 `OpenAISDK` 配置即可：

```python
from opencompass.models import OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )

models = [
    dict(
        abbr='my-openai-compatible-model',
        type=OpenAISDK,
        path='your-model-name',
        key='EMPTY',  # 本地无鉴权服务可用占位值
        meta_template=api_meta_template,
        openai_api_base='http://127.0.0.1:15553/v1',  # 对应你的 API Base
        temperature=0,
        max_out_len=10240,  # 映射到请求中的 max_tokens
        max_seq_len=32768,
        retry=60,  # 请求失败重试次数
        timeout=180,  # 单次请求超时（秒）
        query_per_second=1,
        batch_size=1,
        verbose=True,
        openai_extra_kwargs=dict(img_detail='high'),  # 透传自定义字段
    ),
]
```

参数映射说明：

- `openai_api_base`：OpenAI 兼容服务地址（如 `http://host:port/v1`）
- `timeout`：单次 API 调用超时时间（秒）
- `retry`：请求失败后的最大重试次数
- `openai_extra_kwargs`：除标准参数外的额外字段透传（如 `img_detail`）

### KeyeChat（多 ckpt 聊天评测）示例

如果你希望在同一个评测任务里对多个训练 ckpt 做 API 评测，可以使用 `KeyeChat`（位于 `opencompass.models`）：

```python
from opencompass.models import KeyeChat

KEYE_CKPTS = [
    'istep0000200_2nd_autothink',
    'istep0000400_2nd_autothink',
]

COMMON_CFG = dict(
    type=KeyeChat,
    key='EMPTY',
    api_base='http://127.0.0.1:15553/v1',
    max_tokens=10240,
    max_seq_len=32768,
    temperature=0,
    retry=60,
    timeout=180,
    query_per_second=1,
    batch_size=1,
    img_detail='high',
)

models = [
    dict(abbr=f'keye-{ckpt.replace("_", "-")}', ckpt=ckpt, **COMMON_CFG)
    for ckpt in KEYE_CKPTS
]
```

说明：

- `ckpt` 会映射到底层请求中的模型名（`model/path`）。
- `max_tokens` 会作为 API 侧输出长度上限。
- 你只需修改 `KEYE_CKPTS` 和 `COMMON_CFG`，即可批量评测多个 ckpt。

### 通用单文件配置（`eval_flexible.py`）

如果你希望用一个 JSON 同时配置多模型、多数据集和调度策略，可以使用：

- `examples/eval_flexible.py`
- `examples/eval_flexible.sample.json`（可直接复制后修改）

示例 JSON（通过环境变量 `FLEX_EVAL_CONFIG` 指定）：

```json
{
  "models": [
    {"ref": "gpt_4o_2024_05_13"},
    {"inline": {"class": "OpenAISDK", "abbr": "my-api", "path": "my-model"}}
  ],
  "datasets": [
    {"ref": "aime2025_gen"},
    {"ref": "gpqa_gen/gpqa_datasets"}
  ],
  "scheduler": {
    "infer_strategy": "split",
    "infer_num_worker": 8,
    "infer_num_split": 8,
    "infer_max_num_workers": 8,
    "eval_n": 999999,
    "eval_max_num_workers": 1,
    "watch_interval": 3.0,
    "heartbeat_timeout": 300.0,
    "log_interval": 30.0
  },
  "work_dir": "./outputs/flexible_eval"
}
```

接口约定：

- `models[*].ref`：按 OpenCompass `configs/models` 查找并加载该文件中的 `models` 全量列表。
- `models[*].inline`：内联模型参数；`class` 会映射到 `type`（按 OpenCompass 注册名或完整模块路径解析）。
- `datasets[*].ref`：支持 `name` 或 `name/suffix`，省略 `suffix` 时默认 `_datasets`。
- `scheduler` 省略时使用高吞吐默认值（`split infer + watch eval`）。

运行方式：

```bash
cp examples/eval_flexible.sample.json /tmp/eval_job.json
FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py --dry-run
FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py --mode infer
FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py
```

说明：

- 该方式不需要改 OpenCompass 主流程。
- 当前同时支持两种 schema：
  - 推荐使用顶层 `models`/`datasets` 列表。
  - 兼容历史上常用的 Keye 风格顶层 `model`/`data` map，便于本地 API 服务一键评测。
- 内联模型中若设置 `class: "keyeAPI"` 或 `class: "KeyeFastAPI"`，会自动映射到 `KeyeChat`。你也可以直接写 `class: "KeyeChat"`。
- 内联模型支持 `name` 字段；若未显式设置 `abbr`，会使用 `name` 作为输出目录名（`predictions/<name>/<dataset>.json`）。
- 对于 `KeyeChat` 这类本地 API 模型，如果未显式提供 `ckpt/path`，会默认使用模型条目的名字作为 `ckpt`；因此随机 step 名也可以直接写在配置 key 里。

兼容 schema 示例：

```json
{
  "model": {
    "istep0000200_2nd_autothink": {
      "class": "KeyeFastAPI",
      "max_tokens": 2048,
      "temperature": 0,
      "img_detail": "high",
      "retry": 20,
      "timeout": 180,
      "verbose": true,
      "api_base": "http://127.0.0.1:15553/v1"
    }
  },
  "data": {
    "aime2025": {
      "dataset": "aime2025_gen"
    }
  }
}
```

可复用清单导出工具（完整模型/数据集引用）：

```bash
python tools/export_config_catalog.py --format json --output /tmp/opencompass_catalog.json
python tools/export_config_catalog.py --format markdown --output /tmp/opencompass_catalog.md
```

## 新增第三方模型

新增基于第三方的模型，需要在 `opencompass/models` 下新建 `mymodel.py` 文件，继承 `BaseModel`，并实现  `generate` 方法来进行生成式推理， `get_ppl` 方法来进行判别式推理，以及 `get_token_len` 方法来计算 token 的长度。在定义好之后修改对应配置文件名称即可。

```python
from ..base import BaseModel

class MyModel(BaseModel):

    def __init__(self,
                 pkg_root: str,
                 ckpt_path: str,
                 tokenizer_only: bool = False,
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        ...

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings."""
        pass

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        pass

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs."""
        pass
```
