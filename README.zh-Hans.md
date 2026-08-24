# Charlie alpha

Charlie alpha（模型标识：`Charlie-Alpha-4B`）是支持简体中文、繁体中文和英文的统计程序
选择模型，也提供本地数据分析接口。它基于 Apache-2.0 许可的
[`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)，使用 Apple MLX 4-bit QLoRA
微调。这是衍生模型。

> 发布状态：**Experimental v0.3.0**。DGP-Regret 在冻结 DGP 上降低了 regret，但没有改善
> P-Bench 或 StatQA，信息不足案例也出现明显退步。请将它视为程序选择研究成果，不能视为
> 自动统计分析师。

[繁體中文](README.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md) ·
[DGP-Regret 技术报告](docs/DGP_REGRET.md)

`main` 另含尚未发布的 v0.4 开发流程：受约束统计编译器与有升级／回退门槛的
[DGP-Evolve 自我迭代](docs/DGP_EVOLVE.md)。它不改写上述 v0.3 公开结果。

## 功能

运行时只加载一份 4-bit 底模和一个包含 2,129,920 个参数的 adapter：

- 有数据附件或统计问题时使用 `stats` 路由；其他问题使用相同底模的 `base` 路由。
- adapter 从 28 个训练程序中选择方法。数据代理另有 7 个固定评测程序，生成文本不能作为
  程序执行。
- 分析方案包含估计目标、抽样单位、研究设计、相关结构、缺失机制、方法和诊断字段。
- CSV、TSV、JSON 和 Parquet 保留在本机。Python／R 工具在 macOS 沙箱中运行，禁止联网
  和读取其他用户文件。

`adapter` 仍可作为 API 中 `stats` 路由的兼容别名。数值等价测试确认，base 旁路与独立加载
底模的 logits 最大误差为 `0.0`，恢复 adapter 后的误差也为 `0.0`。

## 冻结评测

三组使用相同提示、工具、temperature 0 和冻结题目。normalized regret 越低越好。

| 模型 | Final DGP regret | 方法正确率 | 无效选择率 |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B 底模 | 0.6727 | 20.83% | 63.33% |
| Hard-label 消融 | 0.7016 | 17.50% | 65.83% |
| DGP-Regret | **0.4437** | **45.00%** | **38.33%** |

DGP-Regret 相对底模降低了 34.04% regret；配对 bootstrap 的平均绝对改善为 0.2290，95% CI
为 `[0.1199, 0.3361]`。无效选择率相对下降 39.47%。在 pilot dev 上，完整方法的 regret
比 hard-label 低 8.54%，通过预先设定的 5% 消融门槛。

| 评测 | 底模 | DGP-Regret | 结果 |
| --- | ---: | ---: | --- |
| 三语方法正确率，英文 | 16.67% | 43.33% | +26.67 点 |
| 三语方法正确率，繁中 | 16.67% | 26.67% | +10.00 点 |
| 三语方法正确率，简中 | 30.00% | 40.00% | +10.00 点 |
| P-Bench Raw / Strict | 0% / 0% | 0% / 0% | 未改善 |
| StatQA exact | 1.00% | 1.00% | 未改善 |
| 信息不足案例 | 43.33% | 0% | 退步 |
| 数学／编程／STEM／一般保留集 | 100% | 100% | 无变化 |

完整汇总、置信区间和门槛判定见
[`reports/stats/evaluation.json`](reports/stats/evaluation.json)。能力门槛未全部通过，因此
v0.3.0 只以 Experimental 发布。

## 安装与使用

需要 Apple Silicon Mac、Python 3.12、`uv`，以及项目锁定的 Pixi Python／R 环境：

```bash
make setup
```

分析本地数据：

```bash
uv run charlie-alpha stats analyze \
  --data survey.csv \
  --question "比较 treatment 和 control 的平均结果，数据来自独立随机分派。" \
  --language zh_Hans \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

多轮对话：

```bash
uv run charlie-alpha stats chat \
  --data survey.csv \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

本地 OpenAI-compatible API 默认只绑定 `127.0.0.1`：

```bash
uv run charlie-alpha stats serve \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Charlie-Alpha-4B",
    "messages":[{"role":"user","content":"检查 treatment 对 outcome 的效果"}],
    "charlie_files":["/absolute/path/survey.csv"],
    "charlie_route":"stats",
    "charlie_tools":true
  }'
```

API 会返回路由、工具次数、隔离状态和分析方案，不返回隐藏推理。

## 已知限制

v0.3 的 DGP 引擎是公开的半参数 operating-characteristic emulator。它使用共同随机数和
128／256／512 次抽样，但没有在每次 replication 中把 28 个方法全部重新拟合到原始表格。
Final DGP 分数衡量模型是否符合这个模拟器，不代表一般统计最优性。

目前数据代理在 P-Bench 几乎总是安全回退为 `needs_clarification`，因此没有产生可计分的
p 值。adapter 也没有学会在信息不足案例中稳定要求补充资料。用户应明确提供估计目标、
抽样单位、配对或聚类结构、分派机制和缺失假设，并检查模型选择的方法。

医疗、政策和财务分析需要合格统计人员审查。GGUF 因 Qwen3.5 hybrid tensor 兼容性尚无
可验证的上游修复而暂不发布；MLX adapter 和融合版已通过干净环境加载。

完整可续跑流程见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。来源 revision 和
许可见 [`configs/sources.lock.json`](configs/sources.lock.json) 与
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。训练文本、缓存、凭据和机器路径不会
提交。v0.2 FORGE 的代码与成果保留在 Git tag `v0.2.0`。

项目代码与模型衍生物采用 [Apache-2.0](LICENSE)；上游数据仍遵循各自许可。
