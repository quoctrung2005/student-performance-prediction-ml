from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# ====== PATH ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(BASE_DIR), "templates")
)

# ====== LOAD MODEL ======
print("📂 Loading models from:", BASE_DIR)

preprocess = joblib.load(os.path.join(BASE_DIR, "preprocess.pkl"))

models = {
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "xgboost_model.pkl")),
    "Linear Regression": joblib.load(os.path.join(BASE_DIR, "linear_regression_model.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "gradient_boosting_model.pkl")),
}

# ====== FEATURE ======
FEATURE_NAMES = [
    "Giờ học mỗi ngày",
    "Giờ sử dụng mạng xã hội",
    "Làm thêm",
    "Tỷ lệ điểm danh (%)",
    "Giờ ngủ",
    "Chất lượng chế độ ăn",
    "Tần suất tập thể dục",
    "Đánh giá sức khỏe tinh thần",
    "GPA học kỳ trước",
    "Mức độ căng thẳng",
    "Nguy cơ bỏ học",
    "Thời gian dùng màn hình",
    "Môi trường học tập",
    "Tiếp cận lớp phụ đạo / gia sư",
    "Mức độ hỗ trợ của cha mẹ",
    "Mức độ động lực",
    "Mức độ lo âu khi thi",
    "Phong cách học tập",
    "Điểm quản lý thời gian",
]

CAT_COLS = [
    "Làm thêm",
    "Chất lượng chế độ ăn",
    "Tần suất tập thể dục",
    "Đánh giá sức khỏe tinh thần",
    "Mức độ căng thẳng",
    "Nguy cơ bỏ học",
    "Môi trường học tập",
    "Tiếp cận lớp phụ đạo / gia sư",
    "Mức độ hỗ trợ của cha mẹ",
    "Mức độ động lực",
    "Mức độ lo âu khi thi",
    "Phong cách học tập",
    "Điểm quản lý thời gian",
]

# ====== ROUTE ======
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            selected_model = request.form.get("model")
            if selected_model not in models:
                raise ValueError("❌ Chưa chọn mô hình")

            model = models[selected_model]

            input_data = {}
            for feature in FEATURE_NAMES:
                value = request.form.get(feature)

                if value is None or value.strip() == "":
                    raise ValueError(f"❌ Thiếu giá trị: {feature}")

                input_data[feature] = value

            df_input = pd.DataFrame([input_data])

            # ====== FIX QUAN TRỌNG ======
            # chỉ convert numeric, không đụng categorical
            for col in df_input.columns:
                if col not in CAT_COLS:
                    df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

            # convert categorical
            for col in CAT_COLS:
                df_input[col] = df_input[col].astype(str)

            # check NaN
            if df_input.isna().any().any():
                raise ValueError("❌ Dữ liệu nhập không hợp lệ")

            print("📥 INPUT:")
            print(df_input)

            # ====== PREDICT ======
            X_processed = preprocess.transform(df_input)
            prediction = float(model.predict(X_processed)[0])

            print("🎯 RESULT:", prediction)

        except Exception as e:
            error = str(e)
            print("❌ ERROR:", error)

    return render_template(
        "index.html",
        prediction=prediction,
        selected_model=selected_model,
        error=error,
        form_data=form_data,
    )

# ====== RUN ======
if __name__ == "__main__":
    app.run(debug=True)