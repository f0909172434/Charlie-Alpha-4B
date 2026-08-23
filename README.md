# Charlie alpha

Charlie alpha（模型識別碼：`Charlie-Alpha-4B`）是一個專攻數學與程式設計的繁中、
簡中、英文實驗模型。它以 Apache-2.0 授權的
[`Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
為底模，在 Apple Silicon 上使用 MLX 4-bit QLoRA 微調；這不是從零預訓練的模型。

> 目前狀態：訓練流程建置中，尚未發布權重或宣稱能力提升。品質門檻未通過時，
> `v0.1.0` 會明確標成 Experimental。

[简体中文](README.zh-Hans.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md)

## 一晚訓練設定

這個分支針對 24 GB Apple Silicon 筆電與約 11 小時的總時限設計：

- 600 筆已驗證的英文短軌跡：數學 300、Python 150、C++ 150。
- 最多各 50 筆繁中與簡中教師精煉；教師下載或生成太慢時，各語言至少 10 筆即可繼續。
- 1,024 tokens、rank 8、最後 8 層 Q/V LoRA、單一短試跑與單一正式設定。
- 正式訓練上限 6 小時、1 epoch；OOM 時固定降到 768 tokens，再降到最後 4 層。
- 先產出可用的 MLX adapter；完整 GGUF 與大評測不會擠占核心訓練時間。

## 快速開始

需要 Apple Silicon Mac、macOS、Python 3.12、`uv`，以及約 15 GB 可用空間。

```bash
make setup
make data
make distill
make mix
make pilot
make train
make eval
make export
```

每個重型步驟都有完成指紋或逐筆續跑檔；重跑不會覆蓋相同設定下的有效成果。
資料與模型來源皆鎖定 commit SHA，詳見
[`configs/sources.lock.json`](configs/sources.lock.json) 與
[`DATA_SOURCES.md`](DATA_SOURCES.md)。

互動與本機服務：

```bash
make chat
make serve
```

## 目標與限制

目標範圍包括高中至大學基礎數學、Python/C++、資料結構與演算法。繁簡中樣本由本機
教師模型翻譯及精煉，公式、數字與程式碼必須逐項保存；不合格樣本會捨棄。

一晚設定優先驗證工程可重現性與小型保留集結果，不足以證明全面能力提升。發布狀態、
分組分數、失敗測試與已知限制會記錄在 `reports/`。請勿把模型輸出視為數學證明、
安全程式碼或專業建議的替代品。

## 授權

專案程式碼與模型衍生物預定以 [Apache-2.0](LICENSE) 發布；資料仍受各上游授權約束。
詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

