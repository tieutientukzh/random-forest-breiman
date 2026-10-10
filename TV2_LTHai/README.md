# TV2 (LTHai) – Thực nghiệm Random Forests trên Pima Indians Diabetes

Thư mục này triển khai các thực nghiệm tham khảo bài báo **Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32** trên dữ liệu **Pima Indians Diabetes**: Forest-RI, Forest-RC, AdaBoost, thử nghiệm nhiễu nhãn, OOB/strength/correlation, độ quan trọng và lựa chọn biến.

**Kết quả chính (100 lần chia dữ liệu):** Forest-RI Single Input có sai số test trung bình **23,71%**, Forest-RI Selection **23,73%**; Forest-RC với F=2 là **23,65%** và F=8 là **23,68%**. Đây là **kết quả thực nghiệm của nhóm trên Diabetes**, không phải các giá trị được công bố trong bài báo gốc. Các chênh lệch rất nhỏ chưa đủ để kết luận thuật toán nào vượt trội.

## 1. Dữ liệu

| Thuộc tính | Mô tả |
|---|---|
| Bộ dữ liệu | Pima Indians Diabetes |
| File | `data/diabetes.csv` |
| Kích thước | 768 mẫu, 8 biến đầu vào, 1 nhãn |
| Nhãn | `Outcome`: 0 hoặc 1 |
| Biến đầu vào | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age |
| Nguồn | Bộ dữ liệu Pima Indians Diabetes được sử dụng phổ biến trong học máy; cần bổ sung liên kết nguồn tải thực tế của nhóm trước khi nộp |

## 2. Xử lý dữ liệu

1. `data_loader.py` đọc CSV, kiểm tra tên cột và nhãn, trả về `X`, `y`, `feature_names`.
2. Trong cấu hình hiện tại, coi giá trị **0** ở `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` là thiếu dữ liệu (`ZERO_AS_MISSING=True`). Đây là quy ước tiền xử lý của thực nghiệm, cần thống nhất với nhóm.
3. Mỗi seed từ **0 đến 99** chia ngẫu nhiên **90% train / 10% test**, **không phân tầng**. Với 768 mẫu: 691 train và 77 test.
4. Tính trung vị **chỉ trên train** để điền dữ liệu thiếu cho cả train và test, tránh rò rỉ dữ liệu.
5. Riêng Forest-RC sử dụng `StandardScaler` học trên train rồi biến đổi train/test; Forest-RI và AdaBoost dùng dữ liệu đã điền thiếu, không chuẩn hóa.

## 3. Các file trong `code/`

| File | Nội dung |
|---|---|
| `data_loader.py` | Đọc và kiểm tra dữ liệu Diabetes. |
| `protocol.py` | Hàm `split_and_impute()`: chia train/test và điền median từ train; có tùy chọn xử lý số 0. |
| `diabetes_random_forest.py` | Chạy Forest-RI, Forest-RC, AdaBoost, nhiễu nhãn, OOB, permutation importance và feature subsets; xuất CSV/PNG. |

**Phụ thuộc module của TV1:** chương trình sử dụng trực tiếp `../TV_NPHieu/code/forest_utils.py` và `../TV_NPHieu/code/forest_rc.py` (đường dẫn được tính từ thư mục gốc repository). Không sao chép hoặc sửa hai module này trong TV2. File `TV2_LTHai/code/protocol.py` là bản riêng cho Diabetes, không phải `TV_NPHieu/code/protocol.py`.

## 4. Cách chạy

Yêu cầu Python có các thư viện trong `requirements.txt` (đặt tại thư mục `TV2_LTHai/`). Các lệnh dưới đây chạy từ **thư mục gốc repository** `random-forest-breiman/` trên Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r TV2_LTHai\requirements.txt
.\.venv\Scripts\python.exe TV2_LTHai\code\diabetes_random_forest.py
```

Nếu chưa tạo môi trường ảo:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r TV2_LTHai\requirements.txt
```

Trong `code/diabetes_random_forest.py`:

- `QUICK = True`: chạy kiểm tra nhanh, 2 seed, 5 cây/rừng, AdaBoost 5 cây.
- `QUICK = False`: chạy chính thức, **100 seed, 100 cây/rừng**, AdaBoost 50 cây. Forest-RC có thể chạy lâu.

Kiểm tra loader và protocol riêng (chạy từ thư mục `TV2_LTHai/`):

```powershell
..\.venv\Scripts\python.exe code\data_loader.py
```

`protocol.py` hiện không có lệnh tự kiểm tra khi chạy trực tiếp; việc chia và điền thiếu được gọi từ chương trình chính.

## 5. Quy trình thực nghiệm

- **Forest-RI:** 100 cây, bootstrap, Gini, không giới hạn độ sâu. Thử `F=1` và `F=int(log2(M)+1)=4` với M=8. Cột `Selection` chọn F có OOB error thấp nhất **trên train**, rồi đánh giá test; `Single Input` cố định F=1.
- **Dự đoán:** bỏ phiếu đa số từ dự đoán của từng cây; hòa phiếu chọn lớp 0. OOB error được tính thủ công bằng phiếu của những cây không học mẫu tương ứng.
- **One Tree ở Bảng 2:** một cây `DecisionTreeClassifier` độc lập học trên train, đánh giá trên test. **Khác** chỉ số `one_tree_oob_error` trong file OOB diagnostics (trung bình lỗi OOB từng cây của rừng); cần thống nhất định nghĩa khi so sánh với TV1/paper.
- **Forest-RC:** 100 cây, L=3, F=2 hoặc F=8; chuẩn hóa trước khi học; dùng module Forest-RC của TV1.
- **AdaBoost:** 50 cây, độ sâu cây cơ sở 1, 3 hoặc 5.
- **Nhiễu nhãn:** lật ngẫu nhiên 5% nhãn **train**, giữ test sạch; so sánh Forest-RI F=1 và F=4.
- **OOB diagnostics:** OOB error, one-tree OOB error, strength, correlation và correlation/strength² cho cấu hình F được Selection chọn.
- **Feature subsets và variable importance:** thực hiện trên **seed=0**, không phải trung bình 100 seed.

## 6. Kết quả

Các số dưới đây được tính từ CSV kết quả 100 seed của TV2. Sai số test trình bày bằng **% trung bình** (chưa kèm SE).

### Bảng 2 – Forest-RI, One Tree, AdaBoost (100 seed)

| Phương pháp | Sai số test trung bình |
|---|---:|
| Forest-RI – Selection | 23,73% |
| Forest-RI – Single Input (F=1) | **23,71%** |
| One Tree (CART độc lập) | 29,27% |
| AdaBoost depth=1 | 24,34% |
| AdaBoost depth=3 | 25,01% |
| AdaBoost depth=5 | 25,34% |

### Bảng 3 – Forest-RC (100 seed)

| Cấu hình | Sai số test trung bình |
|---|---:|
| L=3, F=2 | **23,65%** |
| L=3, F=8 | 23,68% |

### Bảng 4 – Ảnh hưởng nhiễu nhãn (100 seed)

| Forest-RI | Nhãn sạch | Nhiễu 5% | Mức tăng |
|---|---:|---:|---:|
| F=1 | 23,71% | 24,09% | +0,38 điểm % |
| F=4 | 23,65% | 24,32% | +0,67 điểm % |

### OOB, strength và correlation (100 seed)

| Chỉ số | Giá trị trung bình |
|---|---:|
| OOB error | 23,87% |
| One-tree OOB error | 31,66% |
| Strength | 0,3672 |
| Correlation | 0,2973 |
| Correlation / Strength² | 2,2010 |

Lưu ý: trung bình của `correlation / strength²` theo từng seed **khác** việc lấy trung bình correlation chia cho bình phương trung bình strength.

### Feature subsets (chỉ seed=0)

| Số biến | Accuracy |
|---|---:|
| 2 | 79,22% |
| 4 | **81,82%** |
| 6 | 79,22% |
| 8 | 79,22% |

Tập 4 biến gồm `Pregnancies`, `Glucose`, `BMI`, `Age`. Kết quả này mang tính minh họa trên một lần chia, không phải đánh giá trung bình 100 lần.

### Các file kết quả trong `results/`

| File | Nội dung |
|---|---|
| `results_diabetes_table2.csv` | 600 dòng: 100 seed × 6 phương pháp |
| `results_diabetes_table3.csv` | 200 dòng: 100 seed × 2 cấu hình Forest-RC |
| `results_diabetes_table4.csv` | 400 dòng: 100 seed × 2 mức nhiễu × 2 giá trị F |
| `diabetes_oob_diagnostics.csv` | 100 dòng: OOB, strength, correlation |
| `feature_subsets.csv` | 4 dòng: các tập con 2/4/6/8 biến trên seed=0 |
| `variable_importance.png` | Biểu đồ permutation importance trên seed=0 |

## 7. Khác biệt và giới hạn so với paper

- Bộ dữ liệu và các con số trên là **Diabetes**, không phải kết quả gốc của Breast Cancer trong bài báo.
- Forest-RI dùng `RandomForestClassifier` của scikit-learn, không phải mã CART gốc của Breiman; Forest-RC dùng hiện thực riêng của nhóm.
- Quy ước coi 0 là giá trị thiếu và điền median là lựa chọn tiền xử lý của nhóm, không mặc nhiên là quy trình của paper.
- `One Tree` ở Bảng 2 đang là CART độc lập trên test, khác định nghĩa one-tree OOB mà TV1 mô tả; cần thống nhất trước khi ghép bảng báo cáo.
- Feature subsets và variable importance chỉ dùng seed=0, nên chưa có độ ổn định qua 100 lần lặp.
- Các khác biệt sai số rất nhỏ giữa cấu hình không nên diễn giải là có ý nghĩa thống kê nếu chưa kiểm định.
- Chưa tuyên bố tái hiện chính xác mọi bảng/hình của Breiman.

## 8. Tài liệu tham khảo

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Pima Indians Diabetes dataset (nguồn tệp CSV cụ thể cần bổ sung theo dữ liệu nhóm sử dụng).
