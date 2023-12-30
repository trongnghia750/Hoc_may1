import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tập tin Excel
data = pd.read_excel('D:/Học máy/Weather Data.xlsx')
print(data.head())

# Chọn biến độc lập và biến phụ thuộc
X = data[['Wind Speed_km/h', 'Visibility_km']]
y = data['Press_kPa']

# Chuyển đổi biến phụ thuộc thành biến nhị phân (ví dụ: nếu áp suất lớn hơn một ngưỡng, đặt là 1, ngược lại là 0)
threshold = 100
y_binary = (y > threshold).astype(int)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy logistic
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán giá trị trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá hiệu suất của mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# In ra kết quả đánh giá
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# Vẽ biểu đồ để so sánh giá trị thực tế và giá trị dự đoán trên tập kiểm tra
plt.scatter(y_test, y_pred, color='black')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='blue', linewidth=2)
plt.title('Hồi quy Logistic')
plt.xlabel('Actual (Binary)')
plt.ylabel('Predicted (Binary)')
plt.show()