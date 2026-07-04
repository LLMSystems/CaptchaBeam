# CaptchaBeam

**針對固定格式驗證碼 OCR 的可自訂受限 CTC 束搜尋（beam search）解碼與 OpenCV 前處理變體。**

處理流程：**多變體 OpenCV 前處理 → OCR 機率矩陣 →
受限 CTC 束搜尋 → 跨變體共識投票。** 這套方法曾將一個實際的
爬蟲專案的精確準確率從 62% 提升到 85%；本套件將核心邏輯打包，
讓你只需設定字元集與長度，即可套用到*你自己的*驗證碼。

```python
from captchabeam import CaptchaBeam

cb = CaptchaBeam()                       # 18 個變體 + 束搜尋 + 共識投票（最佳設定）
print(cb.decode("captcha.png").text)     # -> "XTRR3"
```

設計理念請參閱 [ARCHITECTURE.md](ARCHITECTURE.md)，完整方法論、
GPU 數據、優化歷程與資料集請參閱 [docs/benchmarks.md](docs/benchmarks.md)。

---

## 安裝

```bash
pip install captchabeam[all]      # 核心 + opencv + ddddocr（預設後端）
```

附加套件：`[cv]` 前處理 · `[ddddocr]` 預設後端 · `[fast]` 快速
後端 · `[gpu]` onnxruntime-gpu · `[eval]`/`[all]`。核心束搜尋
解碼器僅需 `numpy`。

---

## 後端（Backends）

選擇一個後端並傳入 `CaptchaBeam(backend=...)`。所有後端產生的結果
完全相同，差異僅在速度。延遲時間為每張圖片、18 變體束搜尋、
300 張圖片保留測試集，Intel i7-12700（詳見 [docs/benchmarks.md](docs/benchmarks.md)）。

### 預設 — `DdddOcrBackend`

```python
from captchabeam import CaptchaBeam

cb = CaptchaBeam()                                   # 預設後端，184 ms/張，85.0% 精確率
```

### 快速（推薦）— `FastDdddOcrBackend`

使用相同的靜態模型，**輸出逐字元完全一致**，但直接接受陣列
輸入，並採用 numpy 原生解碼。速度快約 34%，準確率不受影響。

```python
from captchabeam import CaptchaBeam
from captchabeam.backends import FastDdddOcrBackend

cb = CaptchaBeam(backend=FastDdddOcrBackend())       # 121 ms/張，85.0% 精確率
```

### GPU — `DdddOcrBackend(use_gpu=True)`

```python
cb = CaptchaBeam(backend=DdddOcrBackend(use_gpu=True))   # 需要 onnxruntime-gpu + CUDA/cuDNN
```

> 對此類工作負載而言，GPU **反而較慢**（模型很小、且未使用批次處理）：347 ms/張。
> 請改用 CPU 執行。詳情請見 [docs/benchmarks.md](docs/benchmarks.md)。

### 使用你自己的模型

任何符合 `png_bytes -> {text, confidence, probabilities, charset}` 介面的
可呼叫物件，都可以作為後端使用（`captchabeam.backends.OcrBackend`）——
自訂的 CRNN 或 PaddleOCR 都能直接接上，束搜尋解碼器依然適用。

```python
cb = CaptchaBeam(backend=MyCRNN())
```

### 延遲 / 準確率一覽表

| 後端 | ms/張 | 精確率 |
|---------|-------:|------:|
| `FastDdddOcrBackend`（CPU） | **121** | 85.0% |
| `DdddOcrBackend`（CPU，預設） | 184 | 85.0% |
| `DdddOcrBackend(use_gpu=True)` | 347 | 85.0% |

---

## 解碼設定

### 速度／準確率分級

```python
CaptchaBeam(variants=1,  decoder="native")   # 最快，單一 Otsu（約 72% 精確率）
CaptchaBeam(variants=6,  decoder="native")   # 平衡（約 76%）
CaptchaBeam(variants=18, decoder="beam")     # 最準確，預設值（85%）
```

### 自訂你的驗證碼規格

```python
from captchabeam import CaptchaBeam, DecodeConfig

cb = CaptchaBeam(decode_config=DecodeConfig(charset="0123456789", length=4, beam_size=16))
```

### 自訂前處理變體

```python
from captchabeam.preprocess import VariantPipeline, resize, otsu, pad, CUBIC

cb = CaptchaBeam(variants=[
    VariantPipeline("a", resize(3, CUBIC), otsu()),
    VariantPipeline("b", resize(2, CUBIC), otsu(), pad(2)),
])
```

### 命令列工具（CLI）

```bash
captchabeam decode captcha.png --variants 18 --decoder beam
captchabeam eval     --data data/captcha_holdout_100 --length 5
captchabeam ablation --data data/captcha_holdout_100          # 逐一排除法（leave-one-out）
```

---

## 開發

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest                          # 若缺少 ddddocr/opencv，相關測試會自動略過
python scripts/benchmark.py [--fast] [--gpu]
```

---

## 授權條款

請參閱 [LICENSE](LICENSE)。
