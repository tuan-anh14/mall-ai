# mall-ai — AI Recommendation Service

Dịch vụ gợi ý sản phẩm thông minh cho hệ thống ShopHub, xây dựng bằng **Python 3.10+** và **FastAPI**. Chạy song song với `mall-be` trên cổng **8001**.

## Kiến trúc tổng quan

```
NestJS (mall-be)
    │
    │  POST /recommend   →  Gợi ý cho user
    │  POST /similar     →  Sản phẩm tương tự
    ▼
FastAPI (mall-ai :8001)
    │
    ├── Cấp 2 — Collaborative Filtering (SVD)
    │       "Người mua X cũng mua Y"
    │       Dựa trên hành vi cộng đồng (view, wishlist, order)
    │
    └── Content-Based Similarity
            Cosine similarity trên vector đặc trưng sản phẩm
            (category, brand, price, rating)
```

Khi AI Service không bật, `mall-be` tự động fallback về **Cấp 1** (content-based đơn giản tích hợp sẵn trong NestJS) — hệ thống luôn hoạt động.

---

## Cấu trúc thư mục

```
mall-ai/
├── main.py                 # FastAPI app, điểm khởi chạy
├── config.py               # Cấu hình từ .env (Pydantic Settings)
├── requirements.txt        # Danh sách thư viện
├── .env.example            # Mẫu biến môi trường
│
├── core/
│   ├── data_loader.py      # Kết nối PostgreSQL, tạo ma trận tương tác
│   ├── features.py         # Feature engineering, cosine similarity
│   ├── trainer.py          # Train SVD + lưu model vào disk
│   └── engine.py           # Singleton engine: load model, serve dự đoán
│
├── api/
│   └── routes.py           # Các endpoint FastAPI
│
├── models/                 # Model đã train (*.pkl) — bỏ qua bởi .gitignore
│
└── tests/
    ├── test_engine.py      # Unit test cho feature engineering
    └── test_api.py         # Integration test cho API endpoints
```

---

## Yêu cầu hệ thống

- Python **3.10+**
- PostgreSQL đang chạy (cùng DB với `mall-be`)
- `mall-be` đã chạy và có dữ liệu (view history, orders, wishlist)

---

## Cài đặt & Khởi chạy

### 1. Tạo virtual environment

```bash
cd mall-ai

# Tạo venv (Python 3.11)
python3.11 -m venv .venv

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Kích hoạt (Windows)
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
```

> `DATABASE_URL` phải trỏ đúng đến DB của `mall-be`.

### 4. Khởi chạy server

```bash
# Cách 1 — chạy trực tiếp
python main.py

# Cách 2 — dùng uvicorn (khuyến nghị khi dev)
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Server sẽ chạy tại `http://localhost:8001`.
Tài liệu API (Swagger UI) tại `http://localhost:8001/docs`.

### 5. Train model (bắt buộc lần đầu)

```bash
# Qua API (server phải đang chạy)
curl -X POST http://localhost:8001/retrain

# Hoặc chạy trực tiếp
python -m core.trainer
```

> Model được lưu tại `models/recommendation_model.pkl`. Nếu chưa train, service vẫn khởi động bình thường nhưng sẽ trả về `[]` cho `/recommend`.

---

## Train Model

Model cần được train trước khi có thể trả về kết quả AI. **Bước này bắt buộc sau lần cài đặt đầu tiên.**

### Cách 1 — Qua API (khi server đang chạy)

```bash
curl -X POST http://localhost:8001/retrain
```

### Cách 2 — Chạy trực tiếp

```bash
python -m core.trainer
```

**Quá trình train:**

```
1. Load dữ liệu từ PostgreSQL
   ├── product_view_histories  → 1.0 điểm/lượt xem (tối đa 10)
   ├── wishlist_items          → 3.0 điểm
   └── order_items             → 5.0 điểm/đơn hàng

2. Tạo User-Item Interaction Matrix
   [userId × productId → score tổng hợp]

3. Train SVD (Singular Value Decomposition)
   n_factors=50, n_epochs=20
   → Học các pattern ẩn từ hành vi người dùng

4. Tính Content-Based Similarity Matrix
   → Cosine similarity trên vector [category, brand, price, rating]

5. Lưu vào models/recommendation_model.pkl
```

> **Khi nào nên retrain?** Sau khi có thêm nhiều dữ liệu mới (vài trăm đơn hàng, view history). Có thể schedule chạy `POST /retrain` định kỳ (ví dụ: mỗi đêm).

---

## API Endpoints

### `GET /health`

Kiểm tra trạng thái service.

```json
// Response
{
  "status": "ok",
  "model_ready": true
}
```

---

### `POST /recommend`

Lấy danh sách sản phẩm gợi ý cho một user.

```json
// Request
{
  "userId": "clxyz123abc",
  "limit": 12
}

// Response
{
  "productIds": ["prod_a", "prod_b", "prod_c", ...]
}
```

**Thuật toán:**
1. Dùng SVD để dự đoán điểm số user-product cho toàn bộ catalog
2. Loại trừ sản phẩm user đã xem/mua
3. Trả về top-K có điểm cao nhất

Nếu model chưa train → trả về `[]` (NestJS tự fallback Cấp 1).

---

### `POST /similar`

Lấy danh sách sản phẩm tương tự cho một sản phẩm.

```json
// Request
{
  "productId": "clxyz456def",
  "limit": 8
}

// Response
{
  "productIds": ["prod_x", "prod_y", "prod_z", ...]
}
```

**Thuật toán:** Cosine similarity trên vector đặc trưng sản phẩm (category + brand + price + rating).

---

### `POST /retrain`

Trigger train lại model từ dữ liệu DB hiện tại.

```json
// Response (thành công)
{
  "success": true,
  "message": "Model retrained and reloaded successfully."
}

// Response (lỗi)
{
  "success": false,
  "message": "Not enough data to train model."
}
```

---

## Chạy Tests

```bash
# Chạy toàn bộ test suite
python -m pytest tests/ -v

# Chỉ unit tests (không cần DB)
python -m pytest tests/test_engine.py -v

# Chỉ API tests
python -m pytest tests/test_api.py -v
```

Kết quả mong đợi:

```
tests/test_engine.py::test_get_similar_basic        PASSED
tests/test_engine.py::test_get_similar_excludes_self PASSED
tests/test_engine.py::test_get_similar_exclude_ids  PASSED
tests/test_engine.py::test_get_similar_unknown_product PASSED
tests/test_engine.py::test_build_product_vectors    PASSED
tests/test_api.py::test_health                      PASSED
tests/test_api.py::test_recommend                   PASSED
tests/test_api.py::test_similar                     PASSED
```

---

## Luồng hoạt động đầy đủ

```
User xem sản phẩm
       │
       ▼
mall-fe gọi POST /api/v1/view-history/track
       │
       ▼
mall-be lưu vào bảng product_view_histories
       │
       ▼ (khi user vào HomePage)
mall-fe gọi GET /api/v1/recommendations
       │
       ▼
mall-be: thử gọi POST http://localhost:8001/recommend
       │
       ├── AI available? ──YES──▶ Trả về SVD predictions
       │
       └── AI unavailable? ─NO──▶ Fallback: content-based
                                   (cùng category/brand với lịch sử xem)
```

---

## Tích hợp với mall-be

`mall-be` giao tiếp với `mall-ai` qua HTTP. Cấu hình URL trong `mall-be/.env`:

```env
AI_SERVICE_URL=http://localhost:8001
```

Nếu `mall-ai` không chạy hoặc timeout (>3s), `mall-be` **tự động fallback** về thuật toán Cấp 1 — không có lỗi nào hiển thị ra người dùng.

---

## Thư viện sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| FastAPI | 0.115.6 | Web framework |
| uvicorn | 0.34.0 | ASGI server |
| scikit-surprise | 1.1.4 | SVD Collaborative Filtering |
| scikit-learn | 1.6.0 | Cosine similarity, preprocessing |
| pandas | 2.2.3 | Xử lý dữ liệu |
| numpy | 2.2.1 | Tính toán ma trận |
| SQLAlchemy | 2.0.36 | Kết nối PostgreSQL |
| joblib | 1.4.2 | Serialize/load model |
| pydantic-settings | 2.7.0 | Quản lý config từ .env |

---

## Xử lý sự cố thường gặp

**`scikit-surprise` không cài được:**
```bash
# macOS
xcode-select --install
pip install scikit-surprise

# Linux
sudo apt-get install python3-dev build-essential
pip install scikit-surprise
```

**`Model file not found` khi khởi động:**
Service vẫn khởi động bình thường. Gọi `POST /retrain` hoặc chạy `python -m core.trainer` để tạo model.

**`Not enough interactions to train CF model`:**
Cần ít nhất `MIN_INTERACTIONS=5` dòng trong bảng `product_view_histories`. Tăng số lượng dữ liệu hoặc giảm ngưỡng trong `.env`.

**Kết nối DB thất bại:**
Kiểm tra `DATABASE_URL` trong `.env` — phải khớp với `mall-be/.env`.
