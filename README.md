# Charlie alpha

Charlie alpha（模型識別碼：`Charlie-Alpha-4B`）是一個支援繁中、簡中、英文的數學與
程式實驗模型。它以 Apache-2.0 的
[`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) 為底模，使用 MLX 4-bit QLoRA；
這是衍生微調模型，不是從零預訓練。

> 發布狀態：**Experimental v0.2.0**。兩組彼此不重疊、各 62 題的封存測試都比同條件底模
> 多答對 1 題；但 +1.62 與 +1.61 個百分點都未達預先設定的 +2 點門檻。這是可重現的正向
> 觀察，不足以證明全面或具統計顯著性的優勢。

[简体中文](README.zh-Hans.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md) ·
[FORGE 方法](docs/FORGE.md)

## 推論架構

Charlie alpha 的正式推論形態不是同時載入兩個模型，也不是生成兩次再挑答案。它只載入
一份 4B 底模與一個 8.52 MB adapter，依題目在生成前切換最後 4 層的 8 個 LoRA 模組：

- 中文或程式題：啟用 2,129,920 個 LoRA 參數。
- 英文非程式題：令 LoRA 增量為零，直接使用底模。
- 每題只生成一次；沒有額外教師、評審模型或第二份 4B 權重。
- 數值測試確認關閉 adapter 時與獨立底模的 logits 最大誤差為 `0.0`，重新啟用後與原
  adapter 的最大誤差也為 `0.0`。

這個固定路由是在第一組 final 結果之後提出，之後才建立一組完全不重疊的確認集。確認集
上為 43/62，底模為 42/62；程式領域由 11/16 到 12/16，其餘語言與領域都沒有少答對。
完整證據在 [`reports/v3/evaluation.json`](reports/v3/evaluation.json)。

## 方法與訓練結果

FORGE（Focused One-pass Relative-Gap Gradient Equivalence）把算力集中在高價值更新：

- 9B 教師不重寫大量英文答案，而是與 4B 學生做 teacher-forced 逐 token loss 比較；只學
  學生明顯落後的 token，英文目標保留率為 52.7%。
- 52 個語義群固定為數學 26、Python 13、C++ 13。每群把同題英文、簡中、繁中放在同一次
  梯度累積，再加入 3 筆英文 replay；實際梯度質量精確為 70%／15%／15%。
- 四組等參數短跑比較標準 LoRA、LoRA+ 與選擇性 loss。勝出配方只訓練最後 4 層 rank-32，
  最佳 validation loss 由 0.8640 降至 0.6867。
- final 開封前只在 dev 做 LoRA-B 增量線搜尋，選出 0.22；這不需重訓。

第一組封存 final 的直接 adapter 為 44/62，底模 43/62。它改善程式與中文，但英文 MATH-500
退步，因此沒有把 adapter 永久套在所有問題上。第二組全新確認集檢驗上述稀疏路由，仍多
答對 1 題且沒有分組正確題數下降。兩組測試仍然很小，請勿把結果外推成所有數學與程式
任務都更強。

## 使用

需要 Apple Silicon Mac、Python 3.12 與 `uv`：

```bash
make setup
make forge-router-verify
make forge-chat
```

`make forge-chat` 使用這台機器的訓練成果。只下載公開版本時，可直接提供 Hugging Face
adapter repo，不需重跑訓練：

```bash
uv run charlie-alpha chat --config configs/pipeline.v2.yaml \
  --adapter-path <HF_ACCOUNT>/Charlie-Alpha-4B-MLX-4bit
```

聊天中可用 `/route auto`、`/route base` 或 `/route adapter` 覆寫自動判斷。本機
OpenAI-compatible 非串流端點：

```bash
make forge-serve
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Charlie-Alpha-4B","messages":[{"role":"user","content":"用 Python 寫二分搜尋"}]}'
```

請求可加 `"charlie_route":"base"` 或 `"adapter"`。完整可重現流程、消融、封存與續跑入口見
[`docs/FORGE.md`](docs/FORGE.md)；來源 revision 與授權見
[`configs/sources.lock.json`](configs/sources.lock.json) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。訓練語料、快取、密鑰與機器路徑不會
提交到 GitHub。

## 限制

目前證據只有兩個 62 題小型套件。自動路由是可解釋的高精度規則，仍可能把英文跨領域題
分錯；重要用途可手動覆寫。生成的證明、數值與程式都可能錯，程式應在隔離環境測試。
動態路由無法忠實融合成單一 GGUF，因此 Experimental v0.2.0 以 MLX adapter 與可重現程式
為主要成果，不發布未通過行為等價門檻的 GGUF。

專案程式碼與模型衍生物採 [Apache-2.0](LICENSE)；上游資料仍遵循各自授權。
