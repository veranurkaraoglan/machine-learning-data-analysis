import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# 1. Veri Yükleme
df = pd.read_csv('data.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# 2. PCA ile 2 Boyuta Düşürme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 3. Modelleri Eğitme
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC(kernel='rbf', random_state=42)
knn_model.fit(X_pca, y)
svm_model.fit(X_pca, y)

# 4. Harita Izgarası Oluşturma
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Renkler: 0(İyi Huylu) -> Mavi, 1(Kötü Huylu) -> Kırmızı
cmap_light = ListedColormap(['#a1c9f4', '#ffb482']) # Arka plan (Pastel Mavi, Pastel Kırmızı)
cmap_bold = ListedColormap(['#1f77b4', '#d62728'])   # Noktalar (Koyu Mavi, Koyu Kırmızı)

# 5. Grafikleri Çizme
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- KNN ---
Z_knn = knn_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z_knn, cmap=cmap_light, alpha=0.7)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=60)
axes[0].set_title("KNN Algoritmasının Karar Sınırı", fontsize=14, fontweight='bold')
#axes[0].set_title("KNN Algoritmasının Karar Sınırı\n(Girintili Çıkıntılı ve Ezberci)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Veri Özelliği 1 (PCA)", fontsize=11)
axes[0].set_ylabel("Veri Özelliği 2 (PCA)", fontsize=11)

# --- SVM ---
Z_svm = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z_svm, cmap=cmap_light, alpha=0.7)
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=60)
axes[1].set_title("SVM Algoritmasının Karar Sınırı", fontsize=14, fontweight='bold')
#axes[1].set_title("SVM Algoritmasının Karar Sınırı\n(Daha Yumuşak ve Genellenebilir)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Veri Özelliği 1 (PCA)", fontsize=11)
axes[1].set_ylabel("Veri Özelliği 2 (PCA)", fontsize=11)

# --- AÇIKLAYICI LEJANT (BİLGİ KUTUSU) EKLİYORUZ ---
# Arka plan renklerinin anlamı
bg_benign = mpatches.Patch(color='#a1c9f4', label='Modelin Kararı: İyi Huylu Bölgesi')
bg_malign = mpatches.Patch(color='#ffb482', label='Modelin Kararı: Kanser Bölgesi')
# Noktaların anlamı
pt_benign = mlines.Line2D([], [], color='#1f77b4', marker='o', linestyle='None', markersize=10, markeredgecolor='k', label='Gerçek Hasta: İyi Huylu')
pt_malign = mlines.Line2D([], [], color='#d62728', marker='o', linestyle='None', markersize=10, markeredgecolor='k', label='Gerçek Hasta: Kanserli')

# Lejantı grafiğin en üstüne yatay olarak yerleştir
fig.legend(handles=[bg_benign, bg_malign, pt_benign, pt_malign], 
           loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=12, frameon=True, shadow=True)

plt.suptitle("Karar Sınırları Analizi", fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.88])
plt.show()