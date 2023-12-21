
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Tạo bộ dữ liệu "noisy_circles"
noisy_circles = datasets.make_circles(n_samples=200, factor=0.5, noise=0.05, random_state=170)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(noisy_circles[0], noisy_circles[1], test_size=0.3, random_state=42)

# Chọn loại kernel là RBF và tạo mô hình SVM
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(X_train, y_train)

# Trực quan hóa dữ liệu và đường biên phân loại 2D và 3D
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Vẽ đường biên phân loại 3D
ax = axes[0]
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], clf.decision_function(X_train), c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', s=50, depthshade=True)
ax.set_title('3D Decision Boundary with RBF Kernel')

# Vẽ biểu đồ dữ liệu gốc 2D
ax = axes[1]
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='binary', edgecolors='k')
ax.set_title('Original 2D Data')

# Vẽ điểm dữ liệu
ax = axes[2]
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, marker='o', edgecolors='k')
# Tạo đường biên quyết định
h = .02  # Kích thước bước cho lưới
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ đường biên quyết định
ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

ax.set_title('SVM Decision Boundary - Noisy Circles')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

plt.show()