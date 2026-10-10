### Thành viên: Lê Thanh Hải

### 1. Giới thiệu

Phần thực nghiệm này triển khai và đánh giá thuật toán Random Forest dựa trên nghiên cứu của Leo Breiman, sử dụng bộ dữ liệu Pima Indians Diabetes.

Mục tiêu:
- Xây dựng Random Forest-RI bằng Python.
- So sánh các phương pháp Selection, Single Input, One Tree và AdaBoost.
- Đánh giá ảnh hưởng của nhiễu nhãn 5%.
- Phân tích Strength, Correlation và tỷ số c/s².
- Đánh giá mức độ quan trọng của các thuộc tính.
- Thử nghiệm mô hình với các tập con thuộc tính.

### 2. Cấu trúc thư mục

```text
random-forest-breiman/
├── README.md
├── requirements.txt
├── TV2_LTHai/
│   ├── diabetes_random_forest.py
│   ├── data/
│   │   └── diabetes.csv
│   └── results/
│       ├── results_diabetes_table2.csv
│       ├── results_diabetes_table4.csv
│       ├── feature_subsets.csv
│       └── variable_importance.png
├── TV3_Bac/
└── TV_NPHieu/
```

### 3. Yêu cầu môi trường

- Python 3.11 trở lên, với phiên bản thư viện tương thích.
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- IPykernel

Cài đặt các thư viện:

```bash
python -m pip install -r requirements.txt
```

### 4. Hướng dẫn chạy

Clone repository:

```bash
git clone https://github.com/tieutientukzh/random-forest-breiman.git
cd random-forest-breiman
```

Tạo môi trường ảo:

```bash
py -m venv .venv
```

Kích hoạt trên Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

Cài đặt thư viện:

```bash
python -m pip install -r requirements.txt
```

Đặt bộ dữ liệu `diabetes.csv` vào `TV2_LTHai/data/`.

Chạy chương trình:

```bash
python TV2_LTHai/diabetes_random_forest.py
```

Trong file Python:
- `QUICK = True`: chạy thử nhanh.
- `QUICK = False`: chạy thực nghiệm đầy đủ.

### 5. Nội dung thực nghiệm

| Cell | Nội dung |
|---|---|
| 1 | Cấu hình và đọc dữ liệu |
| 2 | Cài đặt Random Forest-RI, OOB, Strength và Correlation |
| 3 | Kiểm tra với scikit-learn |
| 4 | Thực nghiệm Bảng 2 |
| 5 | Thực nghiệm Bảng 4 với nhiễu nhãn 5% |
| 6 | Phân tích Strength, Correlation và c/s² |
| 7 | Variable Importance |
| 8 | Thử nghiệm tập con thuộc tính |
| 9 | Forest-RC – chờ tích hợp từ thành viên 3 |
| 10 | Tổng hợp kết quả |

### 6. Kết quả đầu ra

Chương trình lưu kết quả vào `TV2_LTHai/results/`:

- `results_diabetes_table2.csv`: kết quả so sánh các phương pháp.
- `results_diabetes_table4.csv`: kết quả thực nghiệm nhiễu nhãn.
- `feature_subsets.csv`: độ chính xác theo tập con thuộc tính.
- `variable_importance.png`: biểu đồ mức độ quan trọng của các thuộc tính.

Các kết quả hiện tại là kết quả thực nghiệm từ chương trình của nhóm, chưa được xác nhận là tái hiện chính xác các giá trị trong bài báo gốc.

### 7. Trạng thái triển khai

- Random Forest-RI: đã triển khai và chạy thử.
- Thực nghiệm Bảng 2 và Bảng 4: đã chạy với `QUICK = False`.
- Phân tích Variable Importance: đã xuất biểu đồ.
- Forest-RC: chưa tích hợp, chờ module của thành viên 3.

### 8. Repository

GitHub: https://github.com/tieutientukzh/random-forest-breiman

Phần thực nghiệm Diabetes: `TV2_LTHai/`.
