import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tập tin Excel
data = pd.read_excel('D:/Học máy/Weather Data.xlsx')

# Hiển thị thông tin về dữ liệu (các hàng đầu tiên)
print(data.head())

# Chọn biến độc lập và biến phụ thuộc
X = data[['Temp_C']]
y = data['Dew Point Temp_C']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán giá trị trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá hiệu suất của mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In ra kết quả đánh giá
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Vẽ biểu đồ để so sánh giá trị dự đoán và giá trị thực tế trên tập kiểm tra
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Mô hình hồi quy tuyến tính')
plt.xlabel('Độ C')
plt.ylabel('Điểm sương')
plt.show()