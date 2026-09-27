# RedVLA YAML 样例

这里的 `.yml` 都是当前 CLI 支持的完整运行配置，可以直接传给 `redvla validate / evaluate / attack / benign --config`。`.yaml` 与 `.yml` 等价。配置不使用额外的继承、include 或自动加载 `.env` 机制。

## 选择样例

| 文件 | 用途 | 默认任务 / 前置条件 |
| --- | --- | --- |
| [pi0-client.yml](pi0-client.yml) | π0 / π0.5 官方 OpenPI WebSocket 客户端 | 1 个 Goal 场景；`ws://127.0.0.1:8000` |
| [openvla-client.yml](openvla-client.yml) | 在独立环境运行 OpenVLA，HTTP 对接评测端 | 1 个 Spatial 场景；端口 8002 |
| [vla-adapter-client.yml](vla-adapter-client.yml) | 在独立环境运行 VLA-Adapter，HTTP 对接评测端 | 1 个 Goal 场景；端口 8001 |
| [openvla-local.yml](openvla-local.yml) | 同一进程内加载 OpenVLA | Spatial checkpoint；需要设置两个环境变量 |
| [vla-adapter-local.yml](vla-adapter-local.yml) | 同一进程内加载 VLA-Adapter | Goal checkpoint；需要设置两个环境变量 |
| [suite-routed-client.yml](suite-routed-client.yml) | 135 个任务按所属套件选择模型服务 | 四个套件对应四个 endpoint；默认关闭视频 |
| [bounded-attack.yml](bounded-attack.yml) | 有限轮次的物体位置攻击搜索 | 1 个 Goal 场景；最多 3 轮，每轮 300 步；端口 8001 |
| [custom-strategy.yml](custom-strategy.yml) | YAML 加载自定义位置搜索插件 | Goal 模型服务；3 轮、每轮 300 步；使用 examples 插件 |

除全量路由样例外，每份配置默认只展开一个 `(场景, episode, seed)`。语言指令仍来自场景 BDDL；样例里的名称不会覆盖原任务目标。`bounded-attack.yml` 是红队攻击搜索，不是安全解求解器。

## 先验证，再运行

在项目根目录、已安装 red-libero 和仿真依赖的环境中执行：

```bash
# 已有模型服务时：
redvla validate --config configs/examples/vla-adapter-client.yml
redvla probe --model remote --endpoint http://127.0.0.1:8001
redvla evaluate --config configs/examples/vla-adapter-client.yml

```

也可以使用统一复现脚本；指定配置即可运行，red-libero、系统依赖与 Python 版本要求见[主教程](../../README.md)：

```bash
export RED_LIBERO_ROOT=/absolute/path/to/red-libero
bash scripts/setup_env.sh
conda activate redvla
bash scripts/reproduce.sh --config configs/examples/pi0-client.yml
```

`validate` 和 `--dry-run` 不启动仿真、不连接模型、不加载权重。它们检查配置、场景与规则文件并展示展开后的作业；模型依赖、checkpoint 内容和实际服务连接仍在运行时检查。`max_steps` 包含 10 步物理预热。

## Client-server 配套启动命令

**评测端 YAML 与模型端启动命令分开配置。** 当前 `redvla serve` 使用 CLI 参数，不接受 `--config`；不要把这些评测 YAML 传给 `serve`。评测进程运行在 RedVLA 仿真环境，模型服务运行在各自原有的模型环境。

### π0 / π0.5

在 OpenPI checkout 和环境中启动官方服务：

```bash
cd /path/to/openpi
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi0_libero \
  --policy.dir=/path/to/pi0_libero_checkpoint
```

π0.5 使用相匹配的 `pi05_libero` 训练配置与 checkpoint。两者都可以使用 `pi0-client.yml`；该 YAML 的 `type: openpi` 选择协议客户端，实际模型由服务端决定。

```bash
# 回到 RedVLA 评测环境和目录
redvla probe --model openpi --endpoint ws://127.0.0.1:8000
redvla evaluate --config configs/examples/pi0-client.yml
```

### OpenVLA / VLA-Adapter

先在相应模型环境中安装轻量的 RedVLA 包：

```bash
/path/to/model-env/bin/python -m pip install -e /path/to/redvla
```

在 OpenVLA 环境中，部署与 Spatial 样例匹配的模型：

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/openvla-env/bin/python -m redvla serve \
  --model openvla --source-root /path/to/openvla \
  --checkpoint /path/to/openvla-libero-spatial \
  --options '{"unnorm_key":"libero_spatial"}' --port 8002
```

在 VLA-Adapter 环境中，部署与 Goal 样例匹配的模型：

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/adapter-env/bin/python -m redvla serve \
  --model vla-adapter --source-root /path/to/VLA-Adapter \
  --checkpoint /path/to/vla-adapter-libero-goal \
  --options '{"unnorm_key":"libero_goal","num_open_loop_steps":8}' --port 8001
```

以上 `/path/to/...` 都需替换成自己的环境、源码和权重路径。`source_root` 指上游模型源码仓库，`checkpoint` 指权重目录。`unnorm_key` 应与 checkpoint 的统计键一致，不是在远程客户端上设置。两个服务同时运行时，应根据可用显存选择 GPU。

不同机器部署时，修改 YAML 的 `path`；HTTP 服务需按部署需求设置 `--host`。鉴权和网络部署说明见 [模型集成手册](../../docs/model-integration.md)。

## 本地进程模式

当前 Python 环境必须同时满足仿真和对应模型的依赖。先导出变量，再运行本地 YAML：

```bash
# OpenVLA
export OPENVLA_ROOT=/absolute/path/to/openvla
export OPENVLA_CHECKPOINT=/absolute/path/to/openvla-libero-spatial
redvla evaluate --config configs/examples/openvla-local.yml

# VLA-Adapter，在它自己的兼容环境中执行
export VLA_ADAPTER_ROOT=/absolute/path/to/VLA-Adapter
export VLA_CHECKPOINT=/absolute/path/to/vla-adapter-libero-goal
redvla evaluate --config configs/examples/vla-adapter-local.yml
```

缺少 `${VARIABLE}` 所需的变量时，配置加载会直接报错；不会自动寻找权重。本地样例的 `validate` 同样需要先设置这些变量。OpenVLA 和 VLA-Adapter 的上游 Python 包可能冲突，环境不兼容时使用上面的 client-server 样例。

## 全量任务和攻击搜索

`suite-routed-client.yml` 使用已有 `tasks-135.yaml`，通过 `models_by_suite` 分配 24 个 Goal、41 个 Spatial、27 个 Object、43 个 Long 任务。若报告同一模型的全量结果，四个服务应部署该模型对应套件的权重。

```bash
redvla evaluate --config configs/examples/suite-routed-client.yml --dry-run
redvla evaluate --config configs/examples/suite-routed-client.yml --task /goal/ --limit 1
redvla evaluate --config configs/examples/suite-routed-client.yml

redvla attack --config configs/examples/bounded-attack.yml --dry-run
redvla attack --config configs/examples/bounded-attack.yml
```

`evaluate` 总是单次固定场景 rollout；只有 `attack` 使用 `max_iterations`。搜索达到预算或触发规则就结束，不保证找到违规位置。`seeds` 与 `episodes` 做笛卡尔积；全量样例把 seeds 改为 `[0, 7, 42]` 会产生 405 个作业，`--limit` 限制的是展开后的作业数量。

多轮位置更新的完整操作和结果判读见[红队搜索复现指南](../../docs/red-team-reproduction.md)。它区分初始位置直接触发与搜索后触发，并提供独立的场景副本和策略配置。

## 复制和修改

所有相对路径相对于当前 YAML 所在目录解析。因此本目录的资源引用使用 `../../data/...`，全量任务清单使用 `../tasks-135.yaml`。可以在同一目录复制样例再修改；如果把 YAML 移到其他目录，需要相应调整这些路径。

结果写入 `outputs/examples/<样例名>/` 下每次运行新建的子目录。常改字段是 `models.*.path`、本地权重环境变量、`max_steps`、`seeds`、`save_videos` 和任务条目。完整字段说明见 [configuration.md](../../docs/configuration.md)。

## 自定义位置搜索和风险场景

启动 Goal 模型服务后，使用自定义位置搜索策略：

```bash
bash scripts/reproduce.sh --mode attack --config configs/examples/custom-strategy.yml
```

位置插件通过 `defaults.placement_optimizer: module:Class` 和 `placement_options` 配置；示例代码在 `examples/strategies/lateral_search.py`。统一脚本将源码根目录加入 Python 导入路径；直接从其他目录运行 CLI 时，需要自行安装插件或设置 `PYTHONPATH`。

需要独立风险场景时，可生成配套 BDDL/状态、规则及 YAML：

```bash
python scripts/create_scene_example.py --output outputs/custom-scene
bash scripts/reproduce.sh --config outputs/custom-scene/config.yml
```

生成器复制已有场景，不改变物理布局。新增/移动障碍需要重新导出匹配的场景状态。详细流程和违规规则语义见[自定义教程](../../docs/customization.md)。约束检查在重试耗尽后会接受最后一次缩小的候选，不能视为严格无碰撞保证。

也可在评测环境中直接平移已有风险物体并导出新状态：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python scripts/create_scene_example.py \
  --output outputs/moved-knife --offset-xy 0.02 0
bash scripts/reproduce.sh --config outputs/moved-knife/config.yml
```

这会将默认场景第 0 个状态中的刀具沿 X 平移 2 厘米，只导出一个新状态，并写入 `scene-edit.json`；不会修改任务定义或原数据。需结合 rollout 检查物理稳定性和风险效果。
