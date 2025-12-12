import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from feature_scaling import preprocess_data

#  Load dataset
df = pd.read_csv("../data/processed/cleaned_data.csv")

X, y = preprocess_data(df)

#  Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100,      # Tăng số lượng cây lên 200 (mặc định 100)
    max_depth=None,        # Cho cây mọc tự do (hoặc để 20-30) thay vì giới hạn quá thấp
    min_samples_split=5,   # Giảm xuống để cây học chi tiết hơn
    random_state=42,
    n_jobs=-1
)    
model.fit(X_train, y_train)

# 4. Dự đoán
y_train_pred_log = model.predict(X_train)
y_test_pred_log = model.predict(X_test)

# --- BƯỚC QUAN TRỌNG: ĐỔI NGƯỢC LOG VỀ TIỀN THẬT ---
# Dùng np.expm1 để đảo ngược np.log1p
y_train_real = np.expm1(y_train)
y_train_pred_real = np.expm1(y_train_pred_log)

y_test_real = np.expm1(y_test)
y_test_pred_real = np.expm1(y_test_pred_log)

# Tính toán sai số dựa trên TIỀN THẬT cho dễ hình dung
train_r2 = r2_score(y_train, y_train_pred_log)
test_r2 = r2_score(y_test, y_test_pred_log)

train_mae = mean_absolute_error(y_train_real, y_train_pred_real) 
test_mae = mean_absolute_error(y_test_real, y_test_pred_real)

print(f"\n>> KẾT QUẢ LINEAR REGRESSION:")
print(f"   Train: R2={train_r2:.4f}, MAE=${train_mae:,.0f}")
print(f"   Test:  R2={test_r2:.4f}, MAE=${test_mae:,.0f}")

# --- 5. VẼ BIỂU ĐỒ (DÙNG BIẾN REAL VỪA TẠO) ---
plt.figure(figsize=(12, 6))

# Vẽ chấm Xanh (Train) - Dùng _real
plt.scatter(y_train_real, y_train_pred_real, color='blue', alpha=0.3, label='Train (Dữ liệu Học)')

# Vẽ chấm Đỏ (Test) - Dùng _real
plt.scatter(y_test_real, y_test_pred_real, color='red', alpha=0.6, label='Test (Dữ liệu Thi)')


plt.xlabel('Doanh thu Thực tế ($)')
plt.ylabel('Doanh thu Dự đoán ($)')
plt.title('So sánh Thực tế vs Dự đoán (Đơn vị: USD)')
plt.legend()

# Quan trọng: Giữ dòng này để số không bị biến thành 1e9
plt.ticklabel_format(style='plain', axis='both') 
plt.show()
# --- 6. VẼ BIỂU ĐỒ 2: FEATURE IMPORTANCE (Yếu tố quan trọng) ---
importances = model.feature_importances_
indices = np.argsort(importances)[-10:] # Lấy top 10 yếu tố

plt.figure(figsize=(10, 6))
plt.title('Top 10 Yếu tố ảnh hưởng nhất (Random Forest)')
plt.barh(range(len(indices)), importances[indices], color='green', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Mức độ quan trọng')
plt.tight_layout()
plt.show()

# Chuẩn bị nội dung cần ghi
log_content = f"""
----------------------------------------
Mô hình:  Random Forest
Training Metrics          Test Metrics
R2 score: {train_r2:.4f}          R2 score: {test_r2:.4f}
MAE: {train_mae:.4f}        MAE: {test_mae:.4f}
"""

# Mở file với chế độ 'a' (Append - Ghi nối tiếp vào đuôi)
# Nếu file chưa có nó sẽ tự tạo, nếu có rồi nó sẽ ghi tiếp xuống dưới
with open("accuracies.txt", "a", encoding="utf-8") as f:
    f.write(log_content)

print("Đã ghi kết quả vào file accuracies.txt")

import joblib
joblib.dump(model, '../web/random_forest_model.joblib')
joblib.dump(X.columns, '../web/model_columns.joblib')