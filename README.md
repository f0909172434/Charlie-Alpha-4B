# Charlie alpha

Charlie alpha（模型識別碼：`Charlie-Alpha-4B`）是一個專攻數學與程式設計的繁中、
簡中、英文實驗模型。它以 Apache-2.0 授權的
[`Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
為底模，在 Apple Silicon 上使用 MLX 4-bit QLoRA 微調；這不是從零預訓練的模型。

> 目前狀態：`Experimental v0.1.0`。60 題同條件小型評測由底模 38.33% 提升至
> 45.00%，但簡中由 80.00% 降至 40.00%，未通過語言退步門檻，不宣稱全面提升。

> `v0.2` 研究分支正在實測 **FORGE**（Focused One-pass Relative-Gap Gradient
> Equivalence）：改用 Qwen3.5-4B 混合架構，以 4B/9B 逐 token 能力差選資料，並把同一題
> 的英文、簡中、繁中版本放在同一次梯度更新中。它只有在凍結後的 62 題全新測試相對同一
> 底模提升至少 2 點，且各語言／領域退步不超過 2 點時，才會宣稱更強。方法、消融與防洩漏
> 設計見 [`docs/FORGE.md`](docs/FORGE.md)。

[简体中文](README.zh-Hans.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md)

## 一晚訓練設定

這個分支針對 24 GB Apple Silicon 筆電與約 11 小時的總時限設計：

- 600 筆已驗證的英文短軌跡：數學 300、Python 150、C++ 150。
- 最多各 50 筆繁中與簡中教師精煉；教師下載或生成太慢時，各語言至少 10 筆即可繼續。
- 1,024 tokens；以兩個 40-iteration 短試跑比較最後 16 層的 rank-8 Q/V 與
  rank-16 Q/K/V/O LoRA，再只用勝出的設定正式訓練。
- 正式訓練上限 6 小時、2 epochs；全 valid set 連續兩次未改善即停止。OOM 時固定降到
  768 tokens，再降到最後 8 層。
- 先產出可用的 MLX adapter；完整 GGUF 與大評測不會擠占核心訓練時間。

## 實際一晚結果

- rank-8 Q/V、最後 16 層勝出；最佳完整驗證 loss 從 1.106 降至 0.586。
- Metal 固定降級用盡後在累計第 490 步資源早停，發布累計第 440 步的最佳檢查點。
- MATH-500 38.46% → 53.85%、GSM8K 66.67% → 75.00%、MBPP+ 0% → 20.00%。
- 英文 32.00% → 44.00%、繁中維持 60.00%、簡中 80.00% → 40.00%。
- adapter、融合 MLX、沙箱、隱私與乾淨環境載入皆通過；GGUF 等價評測延後。

完整數字與限制見 [`reports/evaluation.json`](reports/evaluation.json) 與
[`MODEL_CARD.md`](MODEL_CARD.md)。

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
