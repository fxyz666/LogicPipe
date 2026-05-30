# LogicPipe

一句话简介 — LogicPipe 是一个面向边缘多设备协同 LLM 推理的开源软件项目，提供离线管线规划、分布式 stage 权重加载、依赖感知任务调度和上下文 KV cache 复用能力。

## 项目功能介绍

LogicPipe 用于在多张 GPU 或多台边缘设备上运行协同 LLM 推理。项目会把模型的连续 transformer 层切分成多个 stage，每个 rank 只加载并执行自己的层段权重；推理时各 stage 通过 `torch.distributed` 传递 activation、候选 token 和新生成 token。

在普通单请求生成中，pipeline parallel 容易出现 stage 空闲。LogicPipe 通过结构化 outline 将一个复杂请求拆成多个带依赖关系的 point，并用 DAG scheduler 管理 ready queue。依赖已经满足的 point 会被送入 pipeline 执行；完成后的输出和 contextual KV cache 会被保存，后继 point 启动时可以复用前驱上下文，从而在并行执行和逻辑连贯之间取得平衡。

项目目前支持 prefill、point prefill、Medusa/MBSD decoding、stage 权重切分、离线规划 artifact 复用，以及 `--load_in_4bit` / `--load_in_8bit` 量化加载参数。

## 项目亮点

- 离线管线规划：根据层数、设备数、计算/通信/内存估计规划生成 `PartitionPlan`，并保存为可复用 artifact。
- 多 rank stage 加载：`tools/model_partition.py` 将完整模型权重切成各 rank 的 `stage.bin`，运行时只加载本地层段。
- 依赖感知调度：outline 被解析为 DAG，只有依赖满足的 point 才会进入执行队列。
- 上下文 KV cache 复用：point 完成后导出 contextual cache，后继 point 可以注入前驱上下文。
- Pipeline 内部加速：Medusa/MBSD decoding 直接集成在 pipeline 执行路径中。
- Prefill 管线优化：支持长 prompt 的序列内切片 prefill，减少 prefill 阶段空泡。
- 量化友好：入口保留 `--load_in_4bit` 和 `--load_in_8bit`，方便在显存有限环境中测试。

## 适用场景

- 多 GPU 或多设备上的 LLM pipeline parallel 推理实验。
- 依赖型推理任务的 DAG 调度、上下文复用和并行解码研究。


## 运行模式

- 离线规划模式：`logicpipe.offline.pipeline.OfflinePipelinePlanner` 生成或复用 `artifacts/logicpipe/offline_plan.json`。
- 权重切分模式：`tools/model_partition.py` 根据 `stage_num_hidden_layers_list` 生成各 rank 的 `stage.bin`。
- 多 rank 在线推理模式：多个进程同时运行 `logicpipe_main.py`，每个进程指定不同 `--rank`。
- DAG 协同推理模式：`logicpipe.orchestrator` 生成 outline、注册 point、调度 ready task 并复用上下文 cache。
- 量化测试模式：启动时加入 `--load_in_4bit` 或 `--load_in_8bit`，降低权重加载显存占用。

## 快速开始

### 基础依赖安装

1. 创建并进入 Python 环境。推荐使用 Python 3.10 或更新版本。

```powershell
conda create -n logicpipe python=3.10
conda activate logicpipe
```

2. 通过 `requirements.txt` 安装项目依赖。

```powershell
pip install -r requirements.txt
```

3. 检查 PyTorch 和 Transformers 是否可用。

```powershell
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

### 配置基础模型

1. 将基础模型权重和 Medusa head 权重放到 `model/` 目录下。



2. 修改模型配置文件中的权重路径。

```json
{
  "base_model_name_or_path": "./model/<model-name>/base",
  "medusa_head_path": "./model/<model-name>/medusa_head"
}
```

3. 确认配置中的 pipeline 参数和启动规模一致。

```json
{
  "stage_num_hidden_layers_list": [8, 8, 8, 8],
  "init_method": "tcp://127.0.0.1:23000",
  "distributed_backend": "gloo",
  "device": "cuda"
}
```

### Offline Planning

1. 根据配置文件生成各 rank 的 stage 权重。

```powershell
python tools\model_partition.py --config_file <model-config>
```

2. 检查 stage 权重是否生成。目录名中的 `<model_type>` 由配置文件和 `tools.utils.get_model_type()` 决定。

```powershell
Test-Path temp_<model_type>_world_4_rank_0\stage.bin
Test-Path temp_<model_type>_world_4_rank_1\stage.bin
Test-Path temp_<model_type>_world_4_rank_2\stage.bin
Test-Path temp_<model_type>_world_4_rank_3\stage.bin
```

3. 首次在线启动时，LogicPipe 会生成 `artifacts/logicpipe/offline_plan.json`；之后可通过 `--reuse_offline_artifact` 复用该规划结果。

### Online Inference

当前示例按 4 个 rank 启动，因此建议配置文件中的 `stage_num_hidden_layers_list` 也包含 4 个 stage，并使用 `--world 4 --num_stages 4`。如果配置文件中的 `init_method` 端口被占用，先换成空闲端口。

使用 `logicpipe_main.py` 启动在线推理。先用短问题和 4bit 做 smoke test，分别启动 4 个进程：

```powershell
python logicpipe_main.py --rank 0 --world 4 --num_stages 4 --reuse_offline_artifact --load_in_4bit --question "2+2?"
python logicpipe_main.py --rank 1 --world 4 --num_stages 4 --reuse_offline_artifact --load_in_4bit --question "2+2?"
python logicpipe_main.py --rank 2 --world 4 --num_stages 4 --reuse_offline_artifact --load_in_4bit --question "2+2?"
python logicpipe_main.py --rank 3 --world 4 --num_stages 4 --reuse_offline_artifact --load_in_4bit --question "2+2?"
```

## 仓库结构

```text
.
├── LOGICPIPE.md
│   └── LogicPipe 项目说明文档。
├── requirements.txt
│   └── Python 基础依赖列表。
├── logicpipe_main.py
│   └── CLI 入口，解析 rank/world/config/量化参数并启动协同推理流程。
├── pipeline_inference.py
│   └── 传统 pipeline 推理入口和工具函数调用示例。
├── logicpipe/
│   ├── __init__.py
│   │   └── 导出 LogicPipeOrchestrator 和 LogicPipeResult。
│   ├── orchestrator.py
│   │   └── 端到端编排：离线规划、runtime 构建、outline 生成、DAG 调度、KV cache 注入和解码循环。
│   ├── runner.py
│   │   └── 初始化分布式环境，校验启动参数，加载当前 rank 的 stage 权重。
│   ├── context.py
│   │   └── 保存 args/config/model/tokenizer 等运行时上下文。
│   ├── types.py
│   │   └── 定义 ResourceProfile、PartitionPlan 和 OfflineArtifactMetadata。
│   ├── offline/
│   │   ├── pipeline.py
│   │   │   └── 离线规划入口，串联资源画像、分区求解和 artifact 读写。
│   │   ├── profiler.py
│   │   │   └── 生成计算、通信、内存相关的资源画像。
│   │   ├── partition_dp.py
│   │   │   └── DP 分区求解器，输出连续层段和设备选择结果。
│   │   └── artifact.py
│   │       └── 保存/加载 `artifacts/logicpipe/offline_plan.json` 并校验 metadata。
│   └── online/
│       ├── prefill_engine.py
│       │   └── 封装 skeleton prompt、shared prefix 和 point prompt 的 prefill。
│       ├── dag_scheduler.py
│       │   └── 管理 DAG task、ready queue、dispatch/completion 状态和 controller。
│       └── branch_decoder.py
│           └── Skeleton/branch decoding 抽象，复用 Medusa self-draft decoding。
├── core/
│   ├── __init__.py
│   │   └── 核心 pipeline 包初始化文件。
│   ├── utils.py
│   │   └── prefill、point prefill、normal decoding 和 outline decoding 的高层工具函数。
│   ├── prefilling_pipeline.py
│   │   └── 序列内 prefill 分区和 point saturation 逻辑。
│   ├── decoding_pipeline.py
│   │   └── 管线 decoding step，处理 tree candidates、tree decoding、新 token 同步和 point 完成状态。
│   └── core/
│       ├── communication.py
│       │   └── Prefill 阶段 stage 间 activation 和 sequence length 通信。
│       ├── decoding_communication.py
│       │   └── Decoding 阶段 point-aware 通信。
│       ├── schedules.py
│       │   └── PipelineRuntime，封装 rank 间发送/接收和 activation padding。
│       ├── tag_manager.py
│       │   └── 分布式通信 tag 分配。
│       └── threadsafe_queue.py
│           └── 通信 helper 使用的线程安全队列。
├── tasks/medusa_llama/
│   ├── config/
│   │   └── 基础模型和 Medusa 运行配置文件。
│   ├── llama_config.py
│   │   └── LLaMA/Medusa 配置类和 pipeline stage 参数更新逻辑。
│   ├── medusa_llama_pp.py
│   │   └── Pipeline-parallel Medusa LLaMA 模型实现。
│   ├── kv_cache.py
│   │   └── KV cache 初始化、快照、加载、导出和上下文拼接。
│   ├── outline_decoding_controller.py
│   │   └── 管理 point 请求队列、输入 token、输出文本和完成状态。
│   └── utils.py
│       └── Medusa 解码相关辅助函数。
├── tools/
│   ├── model_partition.py
│   │   └── 将完整基础模型和 Medusa 权重切分为每个 rank 的 `stage.bin`。
│   ├── sot.py
│   │   └── Skeleton-of-Thought prompt、结构化 outline parser 和 point prompt 构造。
│   └── utils.py
│       └── 分布式初始化、模型类型判断、权重加载和保存工具。
├── artifacts/logicpipe/
│   └── offline_plan.json
│       └── 离线规划 artifact，可通过 `--reuse_offline_artifact` 复用。
├── temp_<model_type>_world_<world>_rank_*/
│   └── stage.bin
│       └── 每个 rank 的本地层段权重。
├── docs/
│   └── 设计说明、运行记录和实验材料。
└── tests/
    └── test_pipeline_runtime.py
        └── prefill 通信形状、同步通信和 orchestrator 延迟构造的回归测试。
```
