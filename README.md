# Charlie alpha

Charlie alpha（模型識別碼：`Charlie-Alpha-4B`）是繁體中文、簡體中文與英文的統計程序
選擇模型，也提供本機資料分析介面。它以 Apache-2.0 授權的
[`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) 為底模，使用 Apple MLX 4-bit
QLoRA 微調。這是衍生模型。

> 發布狀態：**Experimental v0.3.0**。DGP-Regret 在封存 DGP 上降低 regret，卻沒有改善
> P-Bench 或 StatQA，資訊不足案例也出現明顯退步。請把它視為程序選擇研究成果，不能視為
> 自動統計分析師。

[简体中文](README.zh-Hans.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md) ·
[DGP-Regret 技術報告](docs/DGP_REGRET.md)

## 功能

執行時只載入一份 4-bit 底模與一個 2,129,920 參數的 adapter：

- 有資料附件或統計問題時使用 `stats` 路由；其他問題使用相同底模的 `base` 路由。
- adapter 從 28 個訓練程序中選法。資料代理另有 7 個固定評測程序，生成文字不能作為程式
  執行。
- 分析方案包含估計目標、抽樣單位、研究設計、相依性、缺失機制、方法與診斷欄位。
- CSV、TSV、JSON 與 Parquet 留在本機。Python／R 工具在 macOS 沙箱中執行，禁止網路與
  存取其他使用者檔案。

`adapter` 仍可作為 API 中 `stats` 路由的相容別名。數值等價測試確認 base 旁路與獨立載入
底模的 logits 最大誤差為 `0.0`，重新啟用 adapter 後的誤差也為 `0.0`。

## 封存評測

三組使用相同提示、工具、temperature 0 與封存題目。normalized regret 越低越好。

| 模型 | Final DGP regret | 方法正確率 | 無效選法率 |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B 底模 | 0.6727 | 20.83% | 63.33% |
| Hard-label 消融 | 0.7016 | 17.50% | 65.83% |
| DGP-Regret | **0.4437** | **45.00%** | **38.33%** |

DGP-Regret 相對底模降低 34.04% regret；配對 bootstrap 的平均絕對改善為 0.2290，95% CI
為 `[0.1199, 0.3361]`。無效選法率相對下降 39.47%。在 pilot dev 上，完整方法的 regret
比 hard-label 低 8.54%，通過預先設定的 5% 消融門檻。

| 評測 | 底模 | DGP-Regret | 結果 |
| --- | ---: | ---: | --- |
| 三語方法正確率，英文 | 16.67% | 43.33% | +26.67 點 |
| 三語方法正確率，繁中 | 16.67% | 26.67% | +10.00 點 |
| 三語方法正確率，簡中 | 30.00% | 40.00% | +10.00 點 |
| P-Bench Raw / Strict | 0% / 0% | 0% / 0% | 未改善 |
| StatQA exact | 1.00% | 1.00% | 未改善 |
| 資訊不足案例 | 43.33% | 0% | 退步 |
| 數學／程式／STEM／一般保留集 | 100% | 100% | 無變化 |

完整彙總、信賴區間與門檻判定見
[`reports/stats/evaluation.json`](reports/stats/evaluation.json)。能力門檻未全部通過，所以
v0.3.0 只以 Experimental 發布。

## 安裝與使用

需要 Apple Silicon Mac、Python 3.12、`uv`，以及專案鎖定的 Pixi Python／R 環境：

```bash
make setup
```

分析本機資料：

```bash
uv run charlie-alpha stats analyze \
  --data survey.csv \
  --question "比較 treatment 與 control 的平均結果，資料是獨立隨機分派。" \
  --language zh_Hant \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

多輪對話：

```bash
uv run charlie-alpha stats chat \
  --data survey.csv \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

本機 OpenAI-compatible API 預設只綁定 `127.0.0.1`：

```bash
uv run charlie-alpha stats serve \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Charlie-Alpha-4B",
    "messages":[{"role":"user","content":"檢查 treatment 對 outcome 的效果"}],
    "charlie_files":["/absolute/path/survey.csv"],
    "charlie_route":"stats",
    "charlie_tools":true
  }'
```

API 會回傳路由、工具次數、隔離狀態與分析方案，不回傳隱藏推理。

## 已知限制

v0.3 的 DGP 引擎是公開的半參數 operating-characteristic emulator。它使用共同亂數與
128／256／512 次抽樣，但沒有在每次 replication 中把 28 個方法全部重新擬合到原始表格。
Final DGP 分數衡量模型是否符合這個模擬器，不代表一般統計最優性。

目前資料代理在 P-Bench 幾乎總是安全回退為 `needs_clarification`，因此沒有產生可計分的
p 值。adapter 也沒有學會在資訊不足案例中穩定要求補充資料。使用者應明確提供估計目標、
抽樣單位、配對或群聚結構、分派機制與缺失假設，並檢查模型選法。

醫療、政策與財務分析需要合格統計人員審查。GGUF 因 Qwen3.5 hybrid tensor 相容性尚未有
可驗證的上游修正而暫不發布；MLX adapter 與融合版已通過乾淨環境載入。

完整可續跑流程見 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。來源 revision 與
授權見 [`configs/sources.lock.json`](configs/sources.lock.json) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。訓練語料、快取、憑證與機器路徑不會
提交。v0.2 FORGE 的程式與成果保留在 Git tag `v0.2.0`。

專案程式碼與模型衍生物採 [Apache-2.0](LICENSE)；上游資料仍遵循各自授權。
