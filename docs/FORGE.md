# Forge：一晚可驗證的三語選擇性蒸餾

[简体中文](FORGE.zh-Hans.md) · [English](FORGE.en.md)

Forge 是 Charlie alpha v0.2 的研究配方。它不是宣稱發明新的基礎模型層，而是把幾個可
獨立驗證的想法組合成一個適合 24 GB Apple Silicon、約一晚預算的訓練系統。成功標準
只有一個：凍結後的全新測試，必須顯示相對同一個 Qwen3.5-4B 底模的實際正確率提升。

## 為何不用「更多資料、更多 epoch」

v0.1 已證明小型 QLoRA 能降低驗證 loss，但也讓簡中保留集大幅退步。問題不是單純資料
不足，而是每筆資料的價值不同、三種語言的梯度互相競爭，以及程式與數學的序列長度差
異會讓名義比例不等於實際更新比例。在固定一晚內，重複餵入低價值樣本只會更快過擬合。

Forge 的依據包括：

- [Rho-1](https://arxiv.org/abs/2404.07965) 的 excess loss：優先學習目前模型比參考模型
  更不熟悉的 token，而不是平均學習所有 token。
- [LESS](https://arxiv.org/abs/2402.04333) 與
  [BIDS](https://arxiv.org/abs/2501.12147) 的資料影響與能力平衡：少量、相關、跨能力平衡
  的資料可比全量微調更有效。
- [LoRA+](https://arxiv.org/abs/2402.12354) 的非對稱學習率：LoRA 的 A、B 矩陣不應被迫
  使用相同更新速度。
- [xCoT](https://arxiv.org/abs/2401.07037) 的跨語言推理對齊，以及
  [STaR](https://arxiv.org/abs/2203.14465) 的可驗證推理軌跡概念。
- [LIMO](https://arxiv.org/abs/2502.03387) 與 [s1](https://arxiv.org/abs/2501.19393)
  顯示高品質、具難度與多樣性的少量推理資料可以非常有效；本專案只把 LIMO 當研究參考，
  因其軌跡在本機的一晚序列預算下過長，不直接混入訓練。

這些論文支持各元件，不代表它們已證明 Forge 的組合一定有效；所以所有組合效益都必須
由等成本消融與鎖定測試判定。

## 方法

1. **更強且更合適的底模。** 使用 Apache-2.0 的 Qwen3.5-4B 4-bit MLX 版本。它有 32 層
   混合架構，每四層由三個 Gated DeltaNet 線性注意力層與一個完整注意力層組成。Adapter
   同時覆蓋兩種層，避免只訓練完整注意力而忽略 75% 的骨幹。
2. **逐 token 教師差距。** 對每筆正確且去污染的資料，用同 tokenizer 的 4B 學生與 9B
   教師各做一次 teacher-forced forward。樣本依正 excess loss、教師信心與多樣性選取；
   訓練只保留高正差距 token，加上每個答案最後至少 32 tokens。這不需要教師生成新的英文
   長答案，因此比一般回應蒸餾快很多。
3. **能力平衡選擇。** 每個 optimizer update 只處理一個能力群；52 群固定為數學 26、
   Python 13、C++ 13。每群八個 microsteps，不以較長的數學答案偷偷增加數學權重。
4. **三語耦合更新。** 每群先放同一題的英文、簡中、繁中三個版本，再放五筆同能力的高
   價值英文 replay。八筆在同一梯度累積內更新，讓兩種中文不是隔很久才補救。每筆 loss
   權重經校正後，梯度質量精確為英文 70%、簡中 15%、繁中 15%。
5. **無損低成本翻譯。** 只請 9B 教師翻譯少量三語錨點；程式碼、LaTeX、URL 與數字先換成
   不可修改的 placeholder，翻譯後再逐一還原並驗證。繁中由合格簡中做保護區段的 OpenCC
   轉換，因此只需一次模型生成。
6. **等參數消融。** 比較最後 16 層 rank-8 標準 LoRA、同形狀 LoRA+、以及全部 32 層
   rank-4 LoRA+。三者可訓練參數必須完全相等；否則程式直接拒絕比較。短跑先看九題鎖定
   三語 canary 的正確率，再以 valid loss 與耗時打破平手；正式配方只訓練一個 epoch，且
   保存初始 adapter 作為「不比底模更差」的 loss 退路。

## 防止自我欺騙

- `configs/evaluation.v2.lock.json` 在訓練前鎖定 34 題 dev 與 62 題 final；兩者和 v0.1 題目
  都不重疊，題目內容以 SHA-256 固定。
- dev 可用於候選選擇；final 在 `forge freeze` 產生配方、資料、提示、adapter 與題鎖雜湊前
  無法執行。凍結後任何一個檔案改變，final 會拒絕執行。
- 數學、程式、英文、簡中、繁中都以相同提示與生成參數比較底模與 Forge。
- 正式發布門檻是總分至少增加 2 個百分點，且任何語言或領域不得下降超過 2 點。未通過時
  只能標為 Experimental；載入、授權、洩漏或沙箱安全失敗時不得發布權重。

## 可重現入口

```bash
make forge-lock
make forge-prepare
make forge-score
make forge-select
make forge-distill
make forge-build
make forge-pilot
make forge-train
make forge-dev
uv run charlie-alpha forge eval --variant qwen35-base --suite dev \
  --config configs/pipeline.v2.yaml
uv run charlie-alpha forge compare --suite dev --config configs/pipeline.v2.yaml
make forge-freeze
make forge-final
```

評分與翻譯逐筆寫入並可續跑；有效的資料、訓練與評測結果均有輸入指紋。final 指令刻意不
提供繞過凍結的選項。
