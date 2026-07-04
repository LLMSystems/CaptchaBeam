# CaptchaBeam 架構文檔

> 可客製化的**限制性 CTC beam search 解碼**與 **OpenCV 前處理 variant** 工具箱。
> `pip install captchabeam`，即可對任意固定規格的驗證碼做「多 variant 前處理 → 機率矩陣 → 限制性 beam 解碼 → 一致性選字」的完整流程。

---

## 1. 專案定位

CaptchaBeam 是從一個既有的實戰爬蟲專案（`reference_file/`，門牌查詢系統的驗證碼自動辨識）中，把**真正可複用、且經過三批 holdout 驗證有效**的辨識核心抽離出來，包裝成一個獨立、框架無關、可 pip 安裝的函式庫。

原專案把所有辨識邏輯都寫死在 `DoorplateScraper` 這個 Selenium 爬蟲類別裡：

- 字元集寫死 `_CAPTCHA_ALLOWED_CHARS = "0-9A-Z"`
- 長度寫死 `captcha_length = 5`
- beam 參數寫死 `_CAPTCHA_BEAM_SIZE = 10`、`_CAPTCHA_BEAM_TOP_CHARS = 8`
- 18 種 variant 寫死在一個 static method 內
- 只支援 `ddddocr` 一種 OCR 後端

CaptchaBeam 把上述全部變成**設定與擴充點**，讓任何人只要知道自己驗證碼的規格（字元集、長度、干擾型態），就能用同一套經驗證的方法論解自己的題目。

### 核心價值主張

從 reference 提煉出的三大可複用能力，全部保留並一般化：

| 能力 | reference 出處 | CaptchaBeam 的一般化 |
|------|----------------|----------------------|
| **限制性 CTC beam search** | `scraper.py:_ctc_beam_decode` | 字元集、長度（定長或範圍）、beam size、top-k 全可設定 |
| **多 OpenCV 前處理 variant** | `scraper.py:_captcha_variant_pngs` | 可組合的 op pipeline + variant registry，內建 18 種、可自訂 |
| **一致性選字（agreement selector）** | `scraper.py:_select_captcha_by_agreement` | 可插拔 selector 策略（agreement / confidence / 自訂） |

另外把原專案的**優化方法論本身**也搬過來當作套件的一部分：評測 harness、資料集、leave-one-out 剪枝實驗（見 §7、§9）。

---

## 2. 頂層設計原則

1. **與爬蟲/瀏覽器完全解耦**：CaptchaBeam 只吃「圖片 bytes / numpy array」，吐「文字 + 信心分數」。不依賴 Selenium。
2. **與 OCR 後端解耦**：透過 `OcrBackend` protocol 接任何能吐出「每時間步機率矩陣 + charset」的模型。內建 `DdddOcrBackend`，但使用者可換自己的 CRNN / PaddleOCR。
3. **設定驅動、可漸進升級**：跟 reference 一樣保留三個層級——單 Otsu（最快）→ 多 variant native（平衡）→ 多 variant beam（最準）。預設值即最佳方案。
4. **延遲載入重依賴**：`opencv-python`、`ddddocr` 為 optional dependency，只有真的用到才 import（沿用 reference 的 lazy import 做法）。
5. **可重現的評測**：套件自帶資料集與評測工具，benchmark 數字可被任何人重跑驗證。

---

## 3. 套件目錄結構

```text
captchabeam/
├── pyproject.toml                 # 打包設定、optional extras（cv / ddddocr / eval）
├── README.md
├── ARCHITECTURE.md                # 本文件
├── src/
│   └── captchabeam/
│       ├── __init__.py            # 公開 API：CaptchaBeam, decode, DecodeConfig...
│       ├── engine.py              # 高階編排器 CaptchaBeam（前處理→OCR→解碼→選字）
│       ├── config.py              # DecodeConfig / PipelineConfig / SelectConfig
│       ├── types.py               # Candidate, DecodeResult, ProbMatrix 等資料型別
│       │
│       ├── preprocess/            # === CV variant 能力 ===
│       │   ├── __init__.py
│       │   ├── ops.py             # 原子操作：resize / otsu / erode / morph / pad
│       │   ├── pipeline.py        # VariantPipeline：把 ops 串成一個具名 variant
│       │   └── presets.py         # 內建 18 種 variant（搬自 reference）+ 子集選擇
│       │
│       ├── decode/               # === 解碼能力 ===
│       │   ├── __init__.py
│       │   ├── base.py            # Decoder protocol
│       │   ├── beam.py            # RestrictedCTCBeamDecoder（核心，搬自 reference）
│       │   ├── greedy.py          # GreedyDecoder（native，對照組）
│       │   └── logmath.py         # logadd 等數值工具
│       │
│       ├── select/              # === 選字能力 ===
│       │   ├── __init__.py
│       │   ├── base.py            # Selector protocol
│       │   ├── agreement.py       # AgreementSelector（搬自 reference）
│       │   └── confidence.py      # ConfidenceSelector（對照組）
│       │
│       ├── backends/             # === OCR 後端抽象 ===
│       │   ├── __init__.py
│       │   ├── base.py            # OcrBackend protocol：__call__(png) -> RawResult
│       │   └── ddddocr_backend.py # DdddOcrBackend（預設）
│       │
│       ├── eval/                 # === 評測方法論（搬自 reference/scripts）===
│       │   ├── __init__.py
│       │   ├── dataset.py         # labels.csv 載入、holdout 管理
│       │   ├── metrics.py         # exact / char / len 指標
│       │   ├── harness.py         # 快取式評測（OCR 結果快取，重跑 selector 不重算）
│       │   └── ablation.py        # leave-one-out variant 剪枝
│       │
│       └── cli.py                 # `captchabeam decode / eval / ablation`
│
├── data/                          # === 搬遷 reference 的評測資料 ===
│   ├── captcha_samples/           # 100 張，初期探索
│   ├── captcha_holdout_100/       # holdout #1
│   ├── captcha_holdout_extra_100/ # holdout #2
│   └── captcha_holdout_extra2_100/# holdout #3（剪枝泛化驗證）
│
├── docs/                          # 搬遷 reference/docs 的圖與優化報告
│   ├── optimization_report.md     # 搬自 OCR_優化報告.md（英文化/一般化）
│   ├── captcha_18_variants_grid.svg
│   ├── restricted_ctc_beam_decoder.png
│   └── ...
│
├── examples/
│   ├── quickstart.py              # pip 後 5 行解一張圖
│   ├── custom_charset.py          # 自訂字元集/長度
│   └── custom_backend.py          # 接自己的 OCR 模型
│
└── tests/
    ├── test_beam.py               # 解碼正確性（含 reference 的 XRR3→XTRR3 案例）
    ├── test_pipeline.py
    ├── test_selector.py
    └── test_eval.py
```

---

## 4. 核心模組設計

### 4.1 `preprocess/` — 可組合的 CV variant

reference 把 18 種 variant 寫死成一串 lambda（`scraper.py:590-609`）。CaptchaBeam 拆成**原子 op + 具名 pipeline + preset registry**三層。

```python
# preprocess/ops.py — 原子操作，全部吃 numpy array 吐 numpy array
def to_gray(img): ...
def otsu(img): ...                         # THRESH_BINARY + THRESH_OTSU
def resize(img, scale, interp): ...        # nearest / area / cubic
def erode(img, kernel): ...
def morph(img, op, kernel): ...            # open / close
def pad(img, px, value=255): ...

# preprocess/pipeline.py
@dataclass(frozen=True)
class VariantPipeline:
    name: str
    steps: tuple[Op, ...]                  # 依序套用的 op

    def apply(self, gray: np.ndarray) -> np.ndarray: ...
    def to_png(self, gray: np.ndarray) -> bytes: ...
```

```python
# preprocess/presets.py — 內建 18 種，順序與 reference 完全一致（見 §8 對照）
BUILTIN_VARIANTS: list[VariantPipeline] = [
    variant("otsu",             scale(1, CUBIC), otsu),
    variant("s4_nearest_otsu",  scale(4, NEAREST), otsu),
    variant("s25_erode_rect2",  scale(2.5, CUBIC), otsu, erode(RECT2)),
    ...  # 共 18 種
]

def select_variants(count: int) -> list[VariantPipeline]:
    """取前 N 種，對應 reference 的 --captcha-variants N（1/6/18）。"""
    return BUILTIN_VARIANTS[: max(1, min(count, len(BUILTIN_VARIANTS)))]
```

**客製化擴充點**：使用者可 `VariantPipeline("my_v", scale(2, CUBIC), otsu, pad(3))` 自組，或整個換掉 registry。variant 順序有意義（前面的較通用、較穩），沿用 reference 靠 leave-one-out 排出來的排序。

### 4.2 `backends/` — OCR 後端抽象

reference 直接呼叫 `ddddocr.classification(png, probability=True)` 拿到 `{"text", "confidence", "probabilities", "charset"}`。CaptchaBeam 把這個介面抽象成 protocol：

```python
# backends/base.py
class RawResult(TypedDict):
    text: str
    confidence: float
    probabilities: list[list[list[float]]]  # [T][1][V] 每時間步 V 維機率
    charset: list[str]                       # 索引 → 字元；charset[0] 慣例為 blank

class OcrBackend(Protocol):
    def __call__(self, png: bytes) -> RawResult: ...
```

```python
# backends/ddddocr_backend.py
class DdddOcrBackend:
    def __init__(self, show_ad: bool = False):
        import ddddocr                         # lazy
        self._ocr = ddddocr.DdddOcr(show_ad=show_ad)

    def __call__(self, png: bytes) -> RawResult:
        return self._ocr.classification(png, probability=True)
```

只要能吐出 `probabilities + charset`，任何 CTC-based 模型都能當後端，beam decoder 就能用。這是把 reference 從「只能用 ddddocr」解放出來的關鍵。

### 4.3 `decode/beam.py` — 限制性 CTC beam decoder（核心資產）

直接搬 reference 的 `_ctc_beam_decode` / `_logadd` / `_captcha_allowed_indices`，但把三個 hardcode 常數變成 `DecodeConfig`：

```python
@dataclass(frozen=True)
class DecodeConfig:
    charset: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 允許字元（限制解碼的核心）
    length: int | None = 5           # 定長；None 表示不限長
    length_range: tuple[int, int] | None = None  # 或給範圍 (min, max)
    beam_size: int = 10              # reference: _CAPTCHA_BEAM_SIZE
    top_chars: int = 8               # reference: _CAPTCHA_BEAM_TOP_CHARS（每步只展開前 k 個字元）
    blank_index: int = 0             # charset 中 blank 的位置
```

```python
class RestrictedCTCBeamDecoder:
    def __init__(self, config: DecodeConfig): ...

    def decode(self, probs: ProbMatrix, charset: list[str]) -> DecodeResult:
        """
        對每個時間步：
          1. 只保留 config.charset 內字元的機率（把 backend charset 索引映射過去）
          2. 每步取 top_chars 個候選展開
          3. 標準 CTC prefix beam search（區分 blank / non-blank 結尾機率）
          4. 長度約束：達到 length 後不再擴展新字元
          5. 保留 beam_size 條路徑
        回傳長度優先、log-prob 次之的最佳路徑；
        confidence 以 exp(logp / T) 正規化，讓不同 variant 間可加總比較。
        """
```

**這段是整個套件最有價值的部分**——它就是 reference 從 78.3%（native）拉到 85.0%（beam）的關鍵。經典案例：greedy 會少字輸出 `XRR3`，beam 保留更多路徑找回 `XTRR3`。一般化後，字元集與長度約束變成使用者輸入，方法論不變。

### 4.4 `select/` — 選字策略

多個 variant 各自 decode 後會得到一組 `Candidate`，需要選一個最終答案。

```python
@dataclass(frozen=True)
class Candidate:
    variant_name: str
    text: str
    confidence: float

class Selector(Protocol):
    def select(self, candidates: list[Candidate]) -> Candidate: ...
```

- **`AgreementSelector`**（beam 模式預設，搬自 reference `_select_captcha_by_agreement`）：先照 text 分組，選「長度正確 → 組內信心總和最高 → 組內最高信心」的那組。多個 variant 投同一答案時最穩。
- **`ConfidenceSelector`**（native 模式預設）：直接選「長度正確 → 信心最高」。reference 已證實單純信心 selector 容易選到「高信心但少字」的錯誤結果，故 beam 模式不用它。

### 4.5 `engine.py` — 高階編排器

把上面全部串起來，對應 reference 的 `_ocr_captcha`：

```python
class CaptchaBeam:
    def __init__(
        self,
        backend: OcrBackend | None = None,        # 預設 DdddOcrBackend()
        variants: int | list[VariantPipeline] = 18,
        decoder: Literal["beam", "native"] = "beam",
        decode_config: DecodeConfig = DecodeConfig(),
        selector: Selector | None = None,          # 依 decoder 自動選預設
    ): ...

    def decode(self, image: bytes | np.ndarray | str | Path) -> DecodeResult:
        """
        1. 讀圖 → 灰階
        2. 對每個 variant 產生前處理圖 → backend 取機率矩陣
        3. beam / greedy 解碼成 Candidate
        4. selector 選最終答案
        回傳 DecodeResult(text, confidence, per_variant=[...])
        """
```

快速路徑：`variants=1, decoder="native"` 時走單 Otsu，完全等價 reference 的預設最省資源路徑。

---

## 5. 公開 API 與使用情境

### 最短路徑（最佳方案即預設）

```python
from captchabeam import CaptchaBeam

cb = CaptchaBeam()                     # 預設 = 18 variants + restricted beam + agreement
print(cb.decode("captcha.png").text)   # -> "XTRR3"
```

### 速度／準確率三檔（對應 reference 的三個層級）

```python
CaptchaBeam(variants=1,  decoder="native")   # 最快，單 Otsu，~72% exact
CaptchaBeam(variants=6,  decoder="native")   # 平衡，~76% exact
CaptchaBeam(variants=18, decoder="beam")     # 最準，~85% exact（預設）
```

### 客製化：自己的驗證碼規格

```python
from captchabeam import CaptchaBeam, DecodeConfig

cb = CaptchaBeam(decode_config=DecodeConfig(
    charset="0123456789",              # 純數字驗證碼
    length=4,                          # 4 碼
    beam_size=16,
))
```

### 客製化：接自己的 OCR 模型

```python
class MyCRNN:                          # 實作 OcrBackend protocol
    def __call__(self, png: bytes) -> RawResult: ...

cb = CaptchaBeam(backend=MyCRNN())
```

### 客製化：自訂前處理 variant

```python
from captchabeam.preprocess import VariantPipeline, scale, otsu, pad, CUBIC

cb = CaptchaBeam(variants=[
    VariantPipeline("my_a", scale(3, CUBIC), otsu),
    VariantPipeline("my_b", scale(2, CUBIC), otsu, pad(2)),
])
```

---

## 6. 客製化擴充點總覽

| 想改的東西 | 擴充方式 | protocol / config |
|-----------|---------|-------------------|
| 字元集、長度 | `DecodeConfig(charset=..., length=...)` | — |
| beam 寬度 / top-k | `DecodeConfig(beam_size=..., top_chars=...)` | — |
| 前處理 variant | 傳 `list[VariantPipeline]` 或自組 ops | `Op` |
| OCR 模型 | 傳自訂 backend | `OcrBackend` |
| 選字策略 | 傳自訂 selector | `Selector` |
| 解碼演算法 | 傳自訂 decoder | `Decoder` |

每一層都是 protocol，使用者只需替換想改的那一層，其餘沿用內建最佳實作。

---

## 7. 評測方法論與資料（搬遷 reference 的優化過程）

CaptchaBeam 不只搬程式碼，也把 reference **證明方案有效的整套評測流程**搬過來，讓 benchmark 可被重跑、讓使用者能對自己的資料集重做同樣的優化。

### 資料集（搬自 `reference_file/data/`）

| 資料集 | 數量 | 用途 |
|--------|-----:|------|
| `captcha_samples` | 100 | 初期探索與比較 |
| `captcha_holdout_100` | 100 | holdout #1 |
| `captcha_holdout_extra_100` | 100 | holdout #2 |
| `captcha_holdout_extra2_100` | 100 | holdout #3（驗證剪枝泛化） |

每個資料夾含 `labels.csv`（`filename,label`）。主判準為三批 holdout 合併 300 張。

### 指標（`eval/metrics.py`）

| 指標 | 意義 |
|------|------|
| `exact` | 整串完全正確 |
| `char` | 字元級正確率 |
| `len_ok` | 輸出長度符合規格的比例（reference 的 `len5`，一般化為 `len == config.length`） |

### 評測 harness（`eval/harness.py`，搬自 `scripts/eval_*.py`）

沿用 reference 的關鍵設計：**快取昂貴的 OCR + beam decode 結果**（以 `path|size|mtime|decoder|variants` 為 key），之後重跑 selector / 剪枝實驗時不必重算 OCR。這讓 selector 策略比較可以秒級迭代。

### Ablation（`eval/ablation.py`，搬自 `scripts/eval_beam_ablation.py`）

leave-one-out：每次拿掉一個 variant 重算合併準確率，用來判斷某個 variant 是否有貢獻、是否該剪枝。reference 正是靠這個排出 18 種 variant 的順序、並用第三批 holdout 驗證剪枝不 overfit。

### CLI（`cli.py`）

```bash
captchabeam decode captcha.png --variants 18 --decoder beam
captchabeam eval  --data data/ --variants 18 --decoder beam   # 印 exact/char/len 表
captchabeam ablation --data data/ --variants 18               # leave-one-out
```

---

## 8. 內建 18 種 variant（與 reference 對照）

順序即 `select_variants(N)` 取前 N 種的順序，與 `scraper.py:590-609` 完全一致：

| # | variant | 做法 |
|--:|---------|------|
| 1 | `otsu` | 原尺寸灰階 + Otsu |
| 2 | `s4_nearest_otsu` | 放大 4× nearest + Otsu |
| 3 | `s4_area_otsu` | 放大 4× area + Otsu |
| 4 | `s3_nearest_otsu` | 放大 3× nearest + Otsu |
| 5 | `s3_area_otsu` | 放大 3× area + Otsu |
| 6 | `s25_erode_rect2` | 2.5× cubic + Otsu + 2×2 rect erode |
| 7 | `s2_erode_rect2` | 2× cubic + Otsu + 2×2 rect erode |
| 8 | `s2_erode_cross2` | 2× cubic + Otsu + 2×2 cross erode |
| 9 | `s2_open_rect2` | 2× cubic + Otsu + 2×2 rect opening |
| 10 | `s25_nearest_otsu` | 2.5× nearest + Otsu |
| 11 | `s2_cubic_otsu` | 2× cubic + Otsu |
| 12 | `s3_erode_cross2` | 3× cubic + Otsu + 2×2 cross erode |
| 13 | `s3_pad2` | 3× cubic + Otsu + 補 2px 白邊 |
| 14 | `s125_nearest_otsu` | 1.25× nearest + Otsu |
| 15 | `s25_close_rect2` | 2.5× cubic + Otsu + 2×2 rect closing |
| 16 | `s2_pad2` | 2× cubic + Otsu + 補 2px 白邊 |
| 17 | `s25_erode_rect3` | 2.5× cubic + Otsu + 3×3 rect erode |
| 18 | `s25_open_rect2` | 2.5× cubic + Otsu + 2×2 rect opening |

`variants=6` 取前 6 種即為 reference 的平衡檔。

---

## 9. reference → CaptchaBeam 遷移對照表

| reference（爬蟲內） | CaptchaBeam（獨立套件） | 遷移動作 |
|---------------------|--------------------------|----------|
| `DoorplateScraper._captcha_variant_pngs` | `preprocess/presets.py` + `pipeline.py` | 拆成 ops + 具名 pipeline |
| `_otsu_png` | `preprocess/ops.py: otsu` | 抽為原子 op |
| `_ctc_beam_decode` | `decode/beam.py` | 常數 → `DecodeConfig` |
| `_logadd` | `decode/logmath.py` | 直接搬 |
| `_captcha_allowed_indices` | `decode/beam.py`（charset 映射） | 一般化字元集 |
| `_select_captcha_by_agreement` | `select/agreement.py` | 直接搬 |
| `_ocr_captcha`（含 provider/降級/log） | `engine.py: CaptchaBeam.decode` | 剝除 Selenium、只留辨識 |
| `_CAPTCHA_ALLOWED_CHARS / LENGTH / BEAM_*` | `DecodeConfig` | hardcode → 設定 |
| `ddddocr.DdddOcr` 直呼 | `backends/ddddocr_backend.py` | 抽 backend protocol |
| `scripts/eval_beam_ablation.py` | `eval/ablation.py` | 模組化 + 進 CLI |
| `scripts/eval_variant_selector.py` | `eval/harness.py` | 模組化 |
| `data/captcha_holdout_*` | `data/` | 直接搬 |
| `docs/*.svg / *.png`、`OCR_優化報告.md` | `docs/` | 搬移並一般化敘述 |

**不遷移**（屬爬蟲專屬，非本套件範圍）：Selenium driver、stealth、反爬節流退避、DataTable 分頁、SQLite/CSV 匯出、人工輸入降級 provider。

---

## 10. 優化歷程與 Benchmark（搬自 OCR_優化報告）

這些數字是 CaptchaBeam 預設值的依據，會保留在 `docs/optimization_report.md` 並可用 `captchabeam eval` 重跑。以三批 holdout 合併 300 張為準：

| 階段 | 方案 | 合併 300 exact | char | len_ok |
|------|------|---------------:|-----:|-------:|
| 1 | 原圖直接辨識（baseline） | 62.0% | — | — |
| 2 | 灰階 + Otsu | 72.3% | — | — |
| 3 | 6 variants + native | 76.3% | — | — |
| 3 | 18 variants + native | 78.3% | 92.9% | 92.3% |
| 5 | **18 variants + restricted beam + agreement** | **85.0%** | **96.3%** | **99.7%** |

重試累積成功率 `1-(1-p)^n`（單次 p=85%）：3 次 → 99.66%、5 次 → 99.99%。這是把「單次不完美的辨識器」變成「實務上高可靠自動化」的關鍵論點，套件文件會保留。

> 註：以上為 reference 針對「門牌站 5 碼 A-Z/0-9」驗證碼的數字。換到別的驗證碼時，方法論不變，但絕對數字需用 `captchabeam eval` 對自己蒐集的 holdout 重測。

---

## 11. 打包與依賴

```toml
# pyproject.toml（節錄）
[project]
name = "captchabeam"
requires-python = ">=3.10"
dependencies = ["numpy"]           # 核心極輕

[project.optional-dependencies]
cv      = ["opencv-python"]        # 前處理 variant 需要
ddddocr = ["ddddocr"]             # 預設後端需要
eval    = ["opencv-python", "ddddocr"]
all     = ["opencv-python", "ddddocr"]

[project.scripts]
captchabeam = "captchabeam.cli:main"
```

沿用 reference 的 lazy import：`opencv` / `ddddocr` 只在真正用到的路徑才 import，缺套件時給明確安裝提示（對應 reference `scraper.py:726` 的錯誤訊息設計）。核心 beam decoder 只依賴 stdlib + numpy，可獨立測試。

---

## 12. 測試策略

- **`test_beam.py`**：用手構的機率矩陣驗證 CTC beam 正確性，包含 reference 記錄的 `XRR3 → XTRR3` 少字修復案例；驗證長度約束、字元集限制、beam_size / top_chars 邊界。
- **`test_pipeline.py`**：每個 op 與 18 種 preset 的輸出形狀 / 決定性（同輸入同輸出）。
- **`test_selector.py`**：agreement vs confidence 在「高信心少字」情境下的差異（reference 觀察到的失敗模式）。
- **`test_eval.py`**：metrics 計算、harness 快取命中、ablation 不改變 selector 結果。
- 純函式核心不需 OCR/GPU，CI 可完整跑；需要 ddddocr 的整合測試標記為 optional。

---

## 13. Roadmap

| 階段 | 內容 |
|------|------|
| M1 | 抽離核心：`decode/beam.py` + `preprocess` + `select` + `DdddOcrBackend`，達到 reference 85% 對齊 |
| M2 | eval harness + ablation + CLI + 搬遷資料集，benchmark 可重跑 |
| M3 | 文件、examples、pyproject 發佈到 PyPI |
| M4（延伸）| length 範圍解碼、language model 融合、多後端（PaddleOCR）、GPU 批次解碼 |

---

## 附錄：一句話總結

CaptchaBeam 把 reference 專案裡「靠固定驗證碼規格（字元集 + 長度 + CTC beam search）而非過擬合特定樣本」得到的辨識核心，一般化成任何人都能 pip 安裝、用自己規格套用的工具箱——**方法論可攜、數字需自測**。
