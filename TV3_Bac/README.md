# TV3 (Bac) – Tái hiện Random Forests trên bộ dữ liệu Sonar

Thư mục này tái hiện các thí nghiệm phân loại trên **Sonar, Mines vs. Rocks** trong bài báo:

> Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

Nội dung gồm **Bảng 2 (Forest-RI), Bảng 3 (Forest-RC), Bảng 4 (nhiễu nhãn 5%), Strength/Correlation, Hình 1 và Variable Importance**, đồng thời so sánh với các số liệu của bài báo.

## 1. Dữ liệu

| Thuộc tính | Thông tin |
|---|---|
| Bộ dữ liệu | Sonar, Mines vs. Rocks |
| Nguồn | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) |
| Tệp dữ liệu | `data/sonar.all-data` |
| Số mẫu | 208 |
| Số đặc trưng | 60 đặc trưng số |
| Nhãn | `R` (Rock/đá), `M` (Mine/mìn) |
| Bài toán | Phân loại nhị phân |

Các đặc trưng được đọc từ 60 cột đầu; cột thứ 61 là nhãn. Trong chương trình, `R` được mã hóa thành 0 và `M` thành 1. Bộ dữ liệu không có giá trị thiếu; các đặc trưng được sử dụng trực tiếp, không bắt buộc chuẩn hóa trong thí nghiệm Sonar này.

## 2. Cấu trúc thư mục

```text
TV3_Bac/
├── README.md
├── requirements.txt
├── code/
│   ├── main.py                 # Điều phối các thí nghiệm và lưu kết quả
│   └── rf_breiman.py           # Forest-RI, Forest-RC, OOB và Variable Importance
├── data/
│   └── sonar.all-data
├── results/
│   ├── table2_forest_ri.csv
│   ├── table3_forest_rc.csv
│   ├── table4_noise.csv
│   ├── strength_correlation.csv
│   ├── figure1_sonar_ri.csv
│   └── variable_importance.csv
└── figures/
    └── hinh1_sonar.png
```

Tệp CSV lưu thông tin chi tiết theo lần lặp/seed; các bảng trung bình và đối chiếu số liệu bài báo được in trên terminal.

## 3. Yêu cầu và cài đặt

Sử dụng **Python 3.10 trở lên** cùng các thư viện trong `requirements.txt`.

Mở Git Bash hoặc terminal tại **thư mục `TV3_Bac/`**, chạy:

```bash
python -m pip install -r requirements.txt
```

> Nếu dùng Python theo đường dẫn riêng trên Windows, có thể thay `python` bằng đường dẫn đến `python.exe` của máy.

## 4. Cách chạy

**Chạy đầy đủ** (100 lần Bảng 2–3, 50 lần Bảng 4, 30 lần mỗi F của Hình 1, 10 lần Variable Importance):

```bash
python code/main.py --jobs 4
```

**Chạy thử nhanh** (giảm số lần lặp, không dùng kết quả này để đối chiếu chính thức với bài báo):

```bash
python code/main.py --quick --jobs 4
```

**Chạy một phần** khi cần:

```bash
python code/main.py --mode tables23 --jobs 4
python code/main.py --mode figure1 --jobs 4
python code/main.py --mode table4 --jobs 4
python code/main.py --mode importance --jobs 4
```

**Lưu ý:** Chế độ `table4` cần các kết quả từ `tables23` và `figure1` để chọn tham số F trên dữ liệu sạch. Hãy chạy hai chế độ đó trước (hoặc dùng `--mode all`, là mặc định). Nếu code trên GitHub của bạn đã được chỉnh tham số dòng lệnh, kiểm tra `python code/main.py --help` để xem danh sách tùy chọn thực tế.

Các thư mục `results/` và `figures/` sẽ được tạo khi chạy nếu chưa tồn tại. Chế độ `--quick` lưu kết quả vào `results_quick/` và `figures_quick/`.

## 5. Thiết lập thực nghiệm

| Thí nghiệm | Phương pháp / cấu hình | Số lần lặp |
|---|---|---:|
| Bảng 2 | Forest-RI: `F=1` và `F=6`; AdaBoost đối chiếu; rừng 100 cây | 100 |
| Bảng 3 | Forest-RC: `L=3`, `F=2` hoặc `F=8`; rừng 100 cây | 100 |
| Bảng 4 | Đảo khoảng 5% nhãn **chỉ trong tập train**; test giữ nhãn sạch | 50 |
| Strength / Correlation | OOB margin, strength `s`, correlation `ρ`, tỷ số `ρ/s²` | Theo Bảng 2–3 |
| Hình 1 | Forest-RI, quét `F=1..50`, mỗi rừng 100 cây | 30 lần/F |
| Variable Importance | Forest-RI, `F=1`, 1.000 cây/rừng, hoán vị đặc trưng theo OOB | 10 |

- Với các bảng phân loại, mỗi lần chia ngẫu nhiên **90% train, 10% test**.
- Mỗi cây huấn luyện trên một mẫu bootstrap có hoàn lại; cây được phát triển không cắt tỉa.
- **Selection** là Test Error của cấu hình F có OOB Error thấp hơn trong cặp thử nghiệm.
- **One Tree** là sai số OOB trung bình của những cây riêng lẻ trong rừng.
- Forest-RC sinh đặc trưng bằng tổ hợp tuyến tính của 3 biến, với hệ số ngẫu nhiên lấy đều trên `[-1, 1]`.
- AdaBoost dùng tối đa 50 cây cơ sở; trong thực nghiệm Sonar, độ sâu tối đa của cây cơ sở là 5.
- Variable Importance được tính bằng mức thay đổi tương đối của OOB Error sau hoán vị; 10 lần chạy × 60 đặc trưng tạo **600 dòng** dữ liệu.

## 6. Kết quả chính và đối chiếu bài báo

### Bảng 2 – Forest-RI (Test Error, %)

| Chỉ tiêu | Thực nghiệm | Breiman (2001) |
|---|---:|---:|
| AdaBoost | 16.33 | 15.6 |
| Selection | 16.10 | 15.9 |
| Single Input, F=1 | 16.81 | 18.0 |
| One Tree | 33.34 | 31.7 |

### Bảng 3 – Forest-RC (Test Error, %)

| Chỉ tiêu | Thực nghiệm | Breiman (2001) |
|---|---:|---:|
| AdaBoost | 16.33 | 15.6 |
| Selection | 13.38 | 13.6 |
| Two Features, F=2 | 13.71 | 13.8 |
| One Tree | 30.97 | 31.7 |

### Bảng 4 – Ảnh hưởng của 5% nhiễu nhãn

Mức thay đổi **tương đối** của Test Error sau khi thêm nhiễu (đơn vị %):

| Phương pháp | Thực nghiệm | Breiman (2001) |
|---|---:|---:|
| AdaBoost | +16.77 | +15.1 |
| Forest-RI | +21.19 | −6.6 |
| Forest-RC | +16.20 | +4.2 |

Kết quả Bảng 2–3 khá gần bài báo; Bảng 4 có sai lệch đáng kể ở Forest-RI và Forest-RC. Đây là **kết quả thực nghiệm thu được**, không thay đổi để khớp số liệu paper. Các nguyên nhân có thể liên quan đến seed, kích thước Sonar nhỏ, cách xây cây và chi tiết triển khai.

### Hình 1 và Variable Importance

- Hình 1 khảo sát `F=1..50`, 30 lần/F; trong lượt chạy đã báo cáo, `F=8` có Test Error trung bình thấp nhất khoảng **14.92%**.
- Variable Importance đánh giá đủ **60 đặc trưng** trên **10 lần chạy**. Kết quả chi tiết nằm trong `results/variable_importance.csv`.

> **Lưu ý:** Những số liệu ở mục này lấy từ lần thực nghiệm đầy đủ đã báo cáo. Nếu bạn chạy lại với seed hoặc cấu hình khác, kết quả có thể thay đổi. Tệp CSV là nguồn để tính bảng mới nhất.

## 7. Các tệp kết quả

| Tệp | Ý nghĩa |
|---|---|
| `results/table2_forest_ri.csv` | Từng lần chạy Forest-RI, AdaBoost, Selection, Single Input và One Tree |
| `results/table3_forest_rc.csv` | Từng lần chạy Forest-RC, Selection, Two Features và One Tree |
| `results/table4_noise.csv` | Sai số trên train sạch/nhãn nhiễu (đánh giá test sạch), theo phương pháp và lần chạy |
| `results/strength_correlation.csv` | Strength, Correlation và `ρ/s²` |
| `results/figure1_sonar_ri.csv` | Test/OOB Error và các chỉ số theo F, seed |
| `results/variable_importance.csv` | Sai số OOB trước/sau hoán vị của từng biến, từng lần chạy |
| `figures/hinh1_sonar.png` | Đồ thị khảo sát F trên Sonar |

## 8. Khác biệt và giới hạn so với bài báo

- Các lần chia dữ liệu và seed không nhất thiết giống thí nghiệm gốc.
- Forest-RI dùng `RandomForestClassifier` của scikit-learn; Forest-RC tự cài đặt theo nguyên lý của bài báo.
- Hình 1 chạy 30 lần/F thay vì 80 lần/F như bài báo để giảm chi phí tính toán.
- Variable Importance trên Sonar được lặp 10 lần để có thể lấy trung bình, thay vì chỉ một lần như thiết lập mô tả trong bài báo.
- Kết quả Bảng 4 có thể biến động mạnh vì Sonar chỉ có khoảng 21 mẫu test/lần.

## 9. Tài liệu tham khảo

1. Breiman, L. (2001). *Random Forests*. *Machine Learning*, 45(1), 5–32.
2. UCI Machine Learning Repository. *Connectionist Bench (Sonar, Mines vs. Rocks)*: https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
