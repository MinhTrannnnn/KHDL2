# 🎬 Hệ Gợi ý Phim Hybrid - Streamlit App

Hệ thống gợi ý phim sử dụng phương pháp Hybrid kết hợp Collaborative Filtering và Content-Based Filtering.

## 📋 Yêu cầu

- Python 3.8+
- Các thư viện trong `requirements.txt`

## 🚀 Cài đặt

1. **Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```

2. **Train và lưu models:**
```bash
python train_models.py
```

Script này sẽ:
- Load và làm sạch dữ liệu từ `data/movies.csv` và `data/ratings.csv`
- Train Collaborative Filtering model (SVD)
- Tính Content Similarity Matrix
- Lưu tất cả models vào thư mục `models/`

3. **Chạy Streamlit app:**
```bash
streamlit run app.py
```

App sẽ tự động mở trong trình duyệt tại `http://localhost:8501`

## 📁 Cấu trúc Project

```
Film/
├── data/
│   ├── movies.csv          # Dữ liệu phim
│   └── ratings.csv         # Dữ liệu đánh giá
├── notebook/
│   └── test.ipynb          # Notebook phân tích và train
├── models/                 # Thư mục chứa models (tạo sau khi chạy train_models.py)
├── train_models.py         # Script train và lưu models
├── app.py                  # Streamlit app
├── requirements.txt        # Dependencies
└── README.md              # File này
```

## 🎯 Tính năng

### 1. Gợi ý Phim
- Chọn User ID
- Xem top N phim được gợi ý
- Hiển thị predicted rating từ cả 2 phương pháp (CF và CB)

### 2. Tìm kiếm Phim
- Tìm kiếm phim theo tên
- Xem thông tin chi tiết phim
- Dự đoán rating cho phim cụ thể

### 3. Thống kê
- Thống kê tổng quan về dataset
- Xem lịch sử đánh giá của user

## ⚙️ Cài đặt

Trong sidebar, bạn có thể:
- Chọn User ID
- Điều chỉnh số lượng phim gợi ý (5-50)
- Điều chỉnh trọng số giữa Collaborative Filtering và Content-Based Filtering

## 📊 Model

### Collaborative Filtering
- Sử dụng SVD (Singular Value Decomposition)
- 50 components
- Dựa trên lịch sử đánh giá của users

### Content-Based Filtering
- Sử dụng TF-IDF vectorization cho genres
- Cosine similarity giữa các phim
- Dựa trên đặc điểm của phim (thể loại)

### Hybrid
- Kết hợp 2 phương pháp với weighted average
- Mặc định: 60% CF + 40% CB

## 🔧 Troubleshooting

**Lỗi: Không tìm thấy file model**
- Đảm bảo đã chạy `python train_models.py` trước khi chạy app

**Lỗi: Module not found**
- Chạy `pip install -r requirements.txt` để cài đặt dependencies

## 📝 Lưu ý

- Quá trình train model có thể mất vài phút
- Content Similarity Matrix được tính cho tất cả các phim trong train set
- Models được cache để tăng tốc độ load

## 👤 Tác giả

Hệ thống gợi ý hybrid cho MovieLens Dataset

