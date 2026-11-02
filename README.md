# Dự Báo Kết Quả Học Tập Sinh Viên Theo Lối Sống (Machine Learning)

Đồ án Khai phá dữ liệu và Machine Learning nhằm phân tích các thói quen hàng ngày (thời gian học tập, giấc ngủ, mạng xã hội, hoạt động thể chất, chế độ ăn uống...) để dự báo điểm số và kết quả học tập của sinh viên.

---

## 📌 Các Tính Năng Chính
- **Khám phá và Tiền xử lý Dữ liệu (EDA)**: Phân tích phân phối, tương quan giữa các yếu tố lối sống và kết quả học tập; chuẩn hóa dữ liệu và mã hóa biến phân loại.
- **Huấn luyện & Đánh giá Đa Mô Hình**:
  - **Decision Tree Regressor**
  - **Random Forest Regressor**
  - **XGBoost Regressor**
  - **Gradient Boosting Regressor**
  - **Linear Regression**
- **Đánh giá hiệu năng**: Đo lường thông qua các chỉ số $R^2$, RMSE, MAE.
- **Web Demo Flask**: Cung cấp giao diện Web trực quan cho phép người dùng nhập vào các chỉ số lối sống và nhận ngay dự báo điểm số GPA/điểm thi.
- **Tài liệu nghiên cứu**: Đầy đủ báo cáo nghiên cứu dạng văn bản (`reports/BTLML.docx`) và slide thuyết trình (`slides/`).

---

## 🛠️ Công Nghệ Sử Dụng
- **Ngôn ngữ**: Python 3.9+
- **Thư viện Học máy**: Scikit-Learn, XGBoost
- **Xử lý & Trực quan hóa dữ liệu**: Pandas, NumPy, Matplotlib, Seaborn
- **Môi trường & Demo**: Jupyter Notebook, Flask, HTML/CSS

---

## 📂 Cấu Trúc Thư Mục
```
student-performance-prediction-ml/
├── data/                                         # Dữ liệu huấn luyện
├── demo/
│   ├── BTL AI.ipynb                              # Toàn bộ quy trình EDA, xử lý & huấn luyện mô hình
│   ├── demo.py                                   # Ứng dụng Flask Web Demo
│   ├── enhanced_student_habits_performance_dataset.csv
│   ├── decision_tree_model.pkl                   # Checkpoint mô hình Decision Tree
│   ├── gradient_boosting_model.pkl               # Checkpoint mô hình Gradient Boosting
│   ├── xgboost_model.pkl                         # Checkpoint mô hình XGBoost
│   ├── linear_regression_model.pkl               # Checkpoint mô hình Linear Regression
│   └── preprocess.pkl                            # Pipeline tiền xử lý dữ liệu
├── reports/                                      # Báo cáo chi tiết đề tài (BTLML.docx)
├── slides/                                       # Slide trình bày báo cáo
├── templates/                                    # Giao diện HTML của Web Demo
└── requirements.txt                              # Thư viện phụ thuộc
```

> [!NOTE]
> **Lưu ý về file mô hình Random Forest (`random_forest_model.pkl`)**:
> Do file trọng số `random_forest_model.pkl` có dung lượng ~114MB (vượt quá giới hạn 100MB cho phép của GitHub), file này đã được loại trừ khỏi kho lưu trữ. Bạn có thể mở notebook `demo/BTL AI.ipynb` và chạy khối mã huấn luyện để tái sinh file mô hình này chỉ trong vài phút, hoặc sử dụng các mô hình khác đã tích hợp sẵn (`xgboost_model.pkl`, `decision_tree_model.pkl`,...).

---

## 🚀 Hướng Dẫn Khởi Chạy Web Demo
```bash
cd demo
pip install -r ../requirements.txt
python demo.py
```
Truy cập trình duyệt tại địa chỉ `http://127.0.0.1:5000` để thử nghiệm dự báo điểm.
