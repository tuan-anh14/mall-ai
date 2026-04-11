# mall-ai — AI Service

Dịch vụ AI cho hệ thống ShopHub, xây dựng bằng **Python 3.10+** và **FastAPI**. Chạy song song với `mall-be` trên cổng **8001**.

Cung cấp hai module độc lập:
- **Recommendation Engine** — Gợi ý sản phẩm dựa trên hành vi người dùng (SVD + Content-Based)
- **Text Moderation Engine** — Kiểm duyệt bình luận tự động (TF-IDF + Logistic Regression, train tại chỗ, không cần API key)

---

## Kiến trúc tổng quan

```
NestJS (mall-be)
    │
    ├── POST /recommend          →  Gợi ý sản phẩm cho user
    ├── POST /similar            →  Sản phẩm tương tự
    ├── POST /moderate/text      →  Kiểm duyệt bình luận
    └── POST /moderate/retrain   →  Train lại model kiểm duyệt
    ▼
FastAPI (mall-ai :8001)
    │
    ├── Recommendation Engine
    │       Collaborative Filtering (SVD) — "Người mua X cũng mua Y"
    │       Content-Based Similarity — cosine similarity trên đặc trưng sản phẩm
    │
    └── Text Moderation Engine
            Regex pre-filter — blacklist từ khóa cứng (instant)
            TF-IDF Vectorizer (unigram+bigram, 10k features)
            Logistic Regression (SAFE / TOXIC / SPAM)
            ~92% accuracy trên seed dataset 582 mẫu tiếng Việt
```

Khi `mall-ai` không bật hoặc timeout, `mall-be` **tự động fallback** — hệ thống không bao giờ bị gián đoạn vì AI service.

---

## Cấu trúc thư mục

```
mall-ai/
├── main.py                      # FastAPI app, điểm khởi chạy
├── config.py                    # Cấu hình từ .env (Pydantic Settings)
├── requirements.txt             # Danh sách thư viện
│
├── core/
│   ├── data_loader.py           # Kết nối PostgreSQL, tạo ma trận tương tác
│   ├── features.py              # Feature engineering, cosine similarity
│   ├── trainer.py               # Train SVD + lưu recommendation model
│   ├── engine.py                # Singleton: load/serve recommendation model
│   ├── seed_data.py             # 582 mẫu text tiếng Việt có nhãn (SAFE/TOXIC/SPAM)
│   ├── moderation_trainer.py    # Train TF-IDF + Logistic Regression pipeline
│   └── text_moderator.py        # Singleton: load/predict/reload moderation model
│
├── api/
│   └── routes.py                # Tất cả endpoint FastAPI
│
├── models/                      # Model đã train (*.pkl) — bỏ qua bởi .gitignore
│   ├── recommendation_model.pkl
│   └── moderation_model.pkl
│
└── tests/
    ├── test_engine.py           # Unit test cho recommendation engine
    ├── test_api.py              # Integration test cho API
    └── test_moderation.py       # Unit test cho text moderation (11 cases)
```

---

## Yêu cầu hệ thống

- Python **3.10+**
- PostgreSQL đang chạy (cùng DB với `mall-be`)
- `mall-be` đã chạy và có dữ liệu (cho recommendation engine)

> Text moderation **không cần DB** — chạy hoàn toàn local từ seed data.

---

## Cài đặt & Khởi chạy

### 1. Tạo virtual environment

```bash
cd mall-ai
python3.11 -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `scikit-surprise` yêu cầu C compiler.
> - macOS: `xcode-select --install`
> - Linux: `sudo apt-get install python3-dev build-essential`
> - Windows: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### 3. Cấu hình môi trường

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/electro
PORT=8001
HOST=0.0.0.0
MODEL_PATH=./models/recommendation_model.pkl
MIN_INTERACTIONS=5
TOP_K_DEFAULT=12

# Moderation
MODERATION_MODEL_PATH=./models/moderation_model.pkl
MODERATION_THRESHOLD=0.7
```

### 4. Khởi chạy server

```bash
# Cách 1 — chạy trực tiếp
python main.py

# Cách 2 — uvicorn (khuyến nghị khi dev)
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Server tại `http://localhost:8001` · Swagger UI tại `http://localhost:8001/docs`

### 5. Train models (lần đầu)

```bash
# Train recommendation model (cần có dữ liệu trong DB)
curl -X POST http://localhost:8001/retrain

# Train moderation model (chỉ cần seed data — không cần DB)
curl -X POST http://localhost:8001/moderate/retrain
```

> Nếu chưa train moderation model, service vẫn hoạt động bằng **regex fallback** (blacklist từ khóa cứng). Gọi `/moderate/retrain` để nâng cấp lên ML model.

---

## API Endpoints

### Recommendation

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Kiểm tra trạng thái service |
| `POST` | `/recommend` | Gợi ý sản phẩm cho user |
| `POST` | `/similar` | Sản phẩm tương tự |
| `POST` | `/retrain` | Train lại recommendation model |

### Text Moderation

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/moderate/text` | Kiểm duyệt một đoạn văn bản |
| `POST` | `/moderate/retrain` | Train lại moderation model |

---

### `GET /health`

```json
{ "status": "ok", "model_ready": true }
```

---

### `POST /recommend`

```json
// Request
{ "userId": "clxyz123abc", "limit": 12 }

// Response
{ "productIds": ["prod_a", "prod_b", "prod_c"] }
```

---

### `POST /similar`

```json
// Request
{ "productId": "clxyz456def", "limit": 8 }

// Response
{ "productIds": ["prod_x", "prod_y", "prod_z"] }
```

---

### `POST /moderate/text`

Kiểm duyệt một đoạn văn bản. Trả về nhãn `SAFE`, `TOXIC`, hoặc `SPAM` cùng confidence score.

```json
// Request
{ "text": "Sản phẩm tốt, giao hàng nhanh" }

// Response — bình luận an toàn
{ "allowed": true, "label": "SAFE", "score": 0.89 }
```

```json
// Request
{ "text": "đmm shop lừa đảo khốn nạn" }

// Response — vi phạm
{ "allowed": false, "label": "TOXIC", "score": 0.97 }
```

```json
// Request
{ "text": "Inbox mình để mua giá rẻ hơn 0909123456" }

// Response — spam
{ "allowed": false, "label": "SPAM", "score": 0.91 }
```

**Logic phân loại:**
- Regex pre-filter chạy trước để bắt ngay số điện thoại, URL, ký tự lặp, từ khóa social spam
- Nếu regex không bắt được → ML model (TF-IDF + LR) dự đoán
- Chỉ chặn khi confidence ≥ `MODERATION_THRESHOLD` (mặc định 0.7)

---

### `POST /moderate/retrain`

Train lại moderation model từ seed data (+ dữ liệu thực tế nếu có).

```json
// Response
{ "success": true, "message": "Moderation model retrained. Accuracy: 0.9205" }
```

---

### `POST /retrain`

Train lại recommendation model từ DB.

```json
// Response
{ "success": true, "message": "Model retrained and reloaded successfully." }
```

---

## Text Moderation — Chi tiết

### Cách hoạt động

```
Input text
    │
    ▼
Regex pre-filter (instant, không cần model)
    ├── Số điện thoại: 0xxxxxxxxx, +84xxxxxxxxx → SPAM
    ├── URL: http://, www. → SPAM
    ├── Ký tự lặp ≥11 lần: aaaaaaaaaaaaa → SPAM
    ├── Social SPAM: "inbox mình", "zalo tôi" → SPAM
    └── Blacklist từ tục: dm, vcl, khốn, ngu... → TOXIC
    │
    ▼ (nếu regex không bắt được)
ML Model (TF-IDF + Logistic Regression)
    │
    ▼
{ allowed, label, score }
```

### Seed dataset

`core/seed_data.py` chứa **582 mẫu** tiếng Việt gán nhãn thủ công:

| Label | Số mẫu | Ví dụ |
|-------|--------|-------|
| `SAFE` (0) | 215 | "Sản phẩm tốt, giao hàng nhanh", "Chất lượng ổn với giá tiền" |
| `TOXIC` (1) | 157 | Bình luận chửi bới, xúc phạm, lăng mạ |
| `SPAM` (2) | 210 | Số điện thoại, URL, gibberish, quảng cáo chui |

### Fallback khi chưa train

Nếu `models/moderation_model.pkl` chưa tồn tại → service dùng regex blacklist cứng (tức thì, không cần load gì). Gọi `POST /moderate/retrain` để nâng cấp lên ML model.

### Retrain với dữ liệu thực tế

```python
# Trong moderation_trainer.py
train_and_save(
    model_path="./models/moderation_model.pkl",
    extra_texts=["bình luận thực tế 1", "bình luận thực tế 2"],
    extra_labels=[0, 1],  # 0=SAFE, 1=TOXIC, 2=SPAM
)
```

Admin có thể trigger retrain qua `POST /admin/moderation/retrain` trên `mall-be`.

---

## Recommendation Engine — Chi tiết

### Quá trình train

```
1. Load dữ liệu từ PostgreSQL
   ├── product_view_histories  → 1.0 điểm/lượt xem (tối đa 10)
   ├── wishlist_items          → 3.0 điểm
   └── order_items             → 5.0 điểm

2. Tạo User-Item Interaction Matrix [userId × productId → score]

3. Train SVD (n_factors=50, n_epochs=20)
   → Học pattern ẩn từ hành vi người dùng

4. Tính Content-Based Similarity Matrix
   → Cosine similarity trên vector [category, brand, price, rating]

5. Lưu models/recommendation_model.pkl
```

> **Khi nào nên retrain?** Sau khi tích lũy thêm nhiều dữ liệu (vài trăm đơn hàng, view history). Có thể schedule `POST /retrain` định kỳ mỗi đêm.

---

## Chạy Tests

```bash
# Toàn bộ test suite
python -m pytest tests/ -v

# Chỉ moderation tests (không cần DB)
python -m pytest tests/test_moderation.py -v

# Chỉ engine tests
python -m pytest tests/test_engine.py -v
```

Kết quả mong đợi:

```
tests/test_moderation.py::test_safe_text                    PASSED
tests/test_moderation.py::test_safe_text_empty              PASSED
tests/test_moderation.py::test_toxic_text_vn                PASSED
tests/test_moderation.py::test_toxic_explicit               PASSED
tests/test_moderation.py::test_spam_phone                   PASSED
tests/test_moderation.py::test_spam_url                     PASSED
tests/test_moderation.py::test_spam_repeated_chars          PASSED
tests/test_moderation.py::test_safe_review                  PASSED
tests/test_moderation.py::test_safe_neutral                 PASSED
tests/test_moderation.py::test_predict_returns_required_keys PASSED
tests/test_moderation.py::test_train_and_predict            PASSED

tests/test_engine.py::test_get_similar_basic                PASSED
tests/test_engine.py::test_get_similar_excludes_self        PASSED
tests/test_engine.py::test_get_similar_exclude_ids          PASSED
tests/test_engine.py::test_get_similar_unknown_product      PASSED
tests/test_engine.py::test_build_product_vectors            PASSED
```

---

## Tích hợp với mall-be

Cấu hình URL trong `mall-be/.env`:

```env
AI_SERVICE_URL=http://localhost:8001
```

**Recommendation:** `mall-be` gọi `/recommend` và `/similar`, fallback về content-based nội bộ nếu timeout.

**Moderation:** `mall-be` gọi `/moderate/text` trước khi lưu Review/Reply. Nếu timeout (3s) → **cho bình luận qua** (không chặn oan).

Admin retrain moderation qua: `POST /admin/moderation/retrain` trên `mall-be`.

---

## Thư viện sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| FastAPI | 0.115.6 | Web framework |
| uvicorn | 0.34.0 | ASGI server |
| scikit-learn | 1.6.0 | TF-IDF, Logistic Regression, Cosine similarity |
| scikit-surprise | 1.1.4 | SVD Collaborative Filtering |
| pandas | 2.2.3 | Xử lý dữ liệu |
| numpy | 2.2.1 | Tính toán ma trận |
| SQLAlchemy | 2.0.36 | Kết nối PostgreSQL |
| joblib | 1.4.2 | Serialize/load model |
| pydantic-settings | 2.7.0 | Quản lý config từ .env |

---

## Xử lý sự cố thường gặp

**Moderation model chưa có — service vẫn hoạt động?**
Đúng. Service dùng regex fallback khi chưa có `models/moderation_model.pkl`. Gọi `POST /moderate/retrain` để tạo ML model.

**`scikit-surprise` không cài được:**
```bash
# macOS
xcode-select --install && pip install scikit-surprise
# Linux
sudo apt-get install python3-dev build-essential && pip install scikit-surprise
```

**`Model file not found` (recommendation) khi khởi động:**
Service vẫn khởi động. Gọi `POST /retrain` hoặc chạy `python -m core.trainer`.

**`Not enough interactions to train CF model`:**
Cần ít nhất `MIN_INTERACTIONS=5` dòng trong `product_view_histories`. Giảm ngưỡng trong `.env` hoặc thêm dữ liệu.

**Kết nối DB thất bại:**
Kiểm tra `DATABASE_URL` trong `.env` — phải khớp với `mall-be/.env`. Moderation không cần DB nên vẫn hoạt động độc lập.
