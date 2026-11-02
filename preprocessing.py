import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, classification_report

# 1. Veriyi Yükleme (Veri setini 'data.csv' adıyla kaydettiğini varsayıyoruz)
df = pd.read_csv('data.csv')

# 2. Gereksiz Sütunları Temizleme
# 'id' sütunu hastanın kimliğidir, teşhise etkisi yoktur. 
# 'Unnamed: 32' ise Kaggle'dan indirilen dosyada genelde boş gelen hatalı bir sütundur.
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

# 3. Hedef Değişkeni Sayısallaştırma (Encoding)
# M (Malignant - Kötü Huylu) -> 1, B (Benign - İyi Huylu) -> 0 yapıyoruz.
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 4. Bağımlı (y) ve Bağımsız (X) Değişkenleri Ayırma
X = df.drop('diagnosis', axis=1) # Özellikler (Hücre yarıçapı, dokusu vb.)
y = df['diagnosis']              # Hedef (1 veya 0)

# 5. Eğitim ve Test Veri Setini Ayırma (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Normalizasyon (Standartlaştırma)
# Özellikle SVM ve MLP (Sinir Ağları) verilerin aynı ölçekte olmasını ister. 
# Tüm değerleri standart bir aralığa çekiyoruz.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Veri ön işleme tamamlandı! Eğitim seti boyutu:", X_train_scaled.shape)

# ... (önceki kodlarının devamı)

print("\n--- 1. Hedef Değişken (Diagnosis) Dönüşümü ---")
print("M ve B etiketleri algoritmalar için 1 ve 0 yapıldı (İlk 5 satır):")
print(df[['diagnosis']].head())

print("\n--- 2. Normalizasyon Sonrası Özellikler (İlk 5 Satır, İlk 6 Sütun) ---")
# Numpy dizisini okunabilir bir tablo (DataFrame) formatına geri çeviriyoruz
# Sütun isimlerini orijinal X veri setinden alıyoruz
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

# Terminalde çok kalabalık durmaması için ilk 6 sütunu gösteriyoruz
# round(3) ile virgülden sonra sadece 3 basamak göstererek temiz bir görüntü elde ediyoruz
print(X_train_scaled_df.iloc[:, :6].head().round(3))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

print("\n--- 6 Modelin Eğitim ve Karşılaştırma Aşaması ---")

# 1. Tüm Modelleri Tanımlama (Derstekiler + Yeni Eklenenler)
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN (K-Nearest)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP (Yapay Sinir Ağı)": MLPClassifier(max_iter=1000, random_state=42)
}

results = []
confusion_matrices = {}

# 2. Döngü ile Eğitim, Tahmin ve Metrik Hesaplama
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) # 1 (Kötü Huylu) sınıfını doğru bulma oranı
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        "Model": name, 
        "Accuracy (%)": round(acc * 100, 2), 
        "Recall (%)": round(rec * 100, 2)
    })
    confusion_matrices[name] = cm

# 3. Tabloyu Yazdırma
results_df = pd.DataFrame(results).sort_values(by="Recall (%)", ascending=False)
print("\nModel Karşılaştırma Tablosu (Recall'a Göre Sıralı):")
print(results_df.to_string(index=False))

# 4. Karmaşıklık Matrislerini (Confusion Matrix) Görselleştirme
# 2 satır, 3 sütunluk dev bir grafik tablosu oluşturuyoruz
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Modellerin Karmaşıklık Matrisi (Confusion Matrix) Karşılaştırması', fontsize=16, fontweight='bold')

axes = axes.flatten() # Döngüde kolay kullanmak için matrisi düzleştiriyoruz

for i, (name, cm) in enumerate(confusion_matrices.items()):
    # Isı haritası (Heatmap) çizimi
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=['İyi Huylu (0)', 'Kötü Huylu (1)'], 
                yticklabels=['İyi Huylu (0)', 'Kötü Huylu (1)'],
                annot_kws={"size": 14}) # Kutuların içindeki sayıların boyutu
    
    axes[i].set_title(f"{name}", fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Modelin Tahmini', fontsize=12)
    axes[i].set_ylabel('Gerçek Durum', fontsize=12)

# Grafikleri ekranda düzgün yerleştirmek için
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() # Grafiği ekrana bas!

# 5. Random Forest "Feature Importance" (Özellik Önemi) Görselleştirmesi
# Sözlükten eğittiğimiz Random Forest modelini çekiyoruz
rf_model = models["Random Forest"]

# Modelin özelliklere verdiği önem derecelerini alıyoruz
importances = rf_model.feature_importances_

# Sütun isimleri (X.columns) ile önem derecelerini eşleştirip bir tablo yapıyoruz
feature_df = pd.DataFrame({
    'Özellik (Feature)': X.columns,
    'Önem Derecesi': importances
}).sort_values(by='Önem Derecesi', ascending=False)

# Sadece kanseri belirleyen en kritik ilk 10 özelliği çizdirelim
plt.figure(figsize=(12, 8))
sns.barplot(x='Önem Derecesi', y='Özellik (Feature)', data=feature_df.head(10), hue='Özellik (Feature)', palette='magma', legend=False)
plt.title('Random Forest: Kanseri Belirleyen En Kritik 10 Hücresel Özellik', fontsize=16, fontweight='bold')
plt.xlabel('Etki Oranı (Ne kadar yüksekse o kadar kritik)', fontsize=12)
plt.ylabel('Hücre Özellikleri', fontsize=12)
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

print("\n--- BONUS BÖLÜM: Kümeleme (Clustering) Deneyi ---")
print("Algoritmaya hastaların sonuçlarını (diagnosis) vermiyoruz.")
print("Sadece hücre özelliklerini kullanarak benzer hastaları gruplamasını istiyoruz.")

# EKSİK OLAN SATIR: Tüm veriyi (X) K-Means için tek seferde normalleştiriyoruz
X_scaled = scaler.fit_transform(X)

# 1. K-Means Algoritmasını Hazırlama
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# 2. Algoritmayı Sadece Özelliklerle Eğitme ve Küme Etiketlerini Alma
clusters = kmeans.fit_predict(X_scaled)

# 3. PCA (Boyut İndirgeme) ile 30 Özelliği 2 Boyuta Düşürme
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ... (Grafik çizdirme kodları plt.figure() ve sonrası tamamen aynı kalacak)

# 4. Kümeleme Sonuçlarını ve Gerçek Etiketleri Karşılaştırma Grafiği
plt.figure(figsize=(14, 10))

# Grafiğin ana başlığı ve alt başlıkları
plt.suptitle('Kümeleme (Clustering) Deneyi: K-Means vs Gerçek Teşhis', fontsize=18, fontweight='bold')

# Grafiğin görselleştirmesi (Scatter Plot)
# Renkler (c=clusters): K-Means'in oluşturduğu kümeleri (0 ve 1) temsil eder.
# Şekiller (markers): Gerçek teşhis etiketlerini (0=İyi, 1=Kötü) temsil eder.
# Çarpı (X) işareti Kötü Huylu (1) hastaları, Yuvarlak (o) İyi Huylu (0) hastaları gösterir.
for marker, label_name, label_id in [('o', 'İyi Huylu (Gerçek)', 0), ('x', 'Kötü Huylu (Gerçek)', 1)]:
    # Sadece o etikete sahip hastaları filtrele
    mask = (y == label_id)
    # Scatter plot'u çiz
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=clusters[mask], # Renkler K-Means'ten
                marker=marker, # Şekiller gerçek teşhisten
                cmap='viridis', # Renk haritası (Kümeleri ayırmak için)
                s=100, # Nokta boyutu
                edgecolors='k' if marker == 'o' else None, # Yuvarlakların etrafına siyah çizgi
                alpha=0.7, # Yarı saydamlık (Noktalar üst üste gelirse görünsün diye)
                label=label_name)

# Grafiğin x ve y ekseni etiketleri (PCA'nın oluşturduğu sanal eksenler)
plt.xlabel('Temel Bileşen 1 (PC1)', fontsize=14, fontweight='bold')
plt.ylabel('Temel Bileşen 2 (PC2)', fontsize=14, fontweight='bold')

import matplotlib.lines as mlines # Bu kütüphaneyi lejant şekilleri için ekliyoruz

# ... (Grafik çizdirme döngüsü ve x/y etiketleri aynı kalıyor)

# --- 1. Sol Üst Lejant (Gerçek Teşhis - Sadece Şekiller) ---
# K-Means renkleriyle karışmaması için şekilleri nötr (siyah) renkte oluşturuyoruz.
benign_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label='İyi Huylu (Gerçek)')
malignant_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                                 markersize=10, label='Kötü Huylu (Gerçek)')

marker_legend = plt.legend(handles=[benign_marker, malignant_marker], 
                           title='Gerçek Teşhis (Şekiller)', title_fontsize=12, 
                           fontsize=11, loc='upper left')
plt.gca().add_artist(marker_legend) # İlk lejantı grafiğe sabitliyoruz

# --- 2. Sağ Üst Lejant (K-Means Kümeleri - Sadece Renkler) ---
cluster_0_patch = mpatches.Patch(color='#440154', label='Küme 0 (K-Means)') # Viridis moru
cluster_1_patch = mpatches.Patch(color='#fde725', label='Küme 1 (K-Means)') # Viridis sarısı

cluster_legend = plt.legend(handles=[cluster_0_patch, cluster_1_patch], 
                            title='K-Means Kümeleri (Renkler)', title_fontsize=12, 
                            fontsize=11, loc='upper right')

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- EKSTRA GÖRSELLEŞTİRMELER: ALGORİTMALARIN İÇ DÜNYASI ---

# 1. LOGISTIC REGRESSION: Katsayı (Ağırlık) Analizi
lr_model = models["Logistic Regression"]

# Logistic Regression katsayılarını (coef_) alıyoruz
lr_coefficients = lr_model.coef_[0]

# Özellikler ve katsayıları birleştirip DataFrame yapıyoruz
lr_df = pd.DataFrame({
    'Özellik': X.columns,
    'Katsayı (Etki Yönü ve Gücü)': lr_coefficients
})

# Katsayıları büyüklüğüne göre sıralıyoruz
lr_df = lr_df.sort_values(by='Katsayı (Etki Yönü ve Gücü)', ascending=False)

# En çok etki eden pozitif (Kanseri tetikleyen) ve negatif (İyi huyluyu tetikleyen) ilk 5 özelliği alalım
top_features = pd.concat([lr_df.head(5), lr_df.tail(5)])

plt.figure(figsize=(10, 6))
# Katsayı sıfırdan büyükse kırmızı (Kötü Huylu etkisi), küçükse mavi (İyi Huylu etkisi) yapıyoruz
colors = ['red' if c > 0 else 'blue' for c in top_features['Katsayı (Etki Yönü ve Gücü)']]
sns.barplot(x='Katsayı (Etki Yönü ve Gücü)', y='Özellik', data=top_features, palette=colors, hue='Özellik', legend=False)
plt.title('Logistic Regression: Kanseri (Kırmızı) ve İyi Huyluyu (Mavi) En Çok Etkileyen Faktörler', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--') # Sıfır noktasına referans çizgisi
plt.tight_layout()
plt.show()

# ---------------------------------------------------------

# 2. MLP (YAPAY SİNİR AĞI): Öğrenme Eğrisi (Loss Curve)
mlp_model = models["MLP (Yapay Sinir Ağı)"]

plt.figure(figsize=(10, 6))
# mlp_model.loss_curve_ bize her adımda hatanın nasıl düştüğünü verir
plt.plot(mlp_model.loss_curve_, color='purple', linewidth=3)
plt.title('MLP (Yapay Sinir Ağı) Öğrenme Eğrisi (Loss Curve)', fontsize=15, fontweight='bold')
plt.xlabel('İterasyon (Epoch - Öğrenme Adımı)', fontsize=12)
plt.ylabel('Hata Payı (Loss)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.fill_between(range(len(mlp_model.loss_curve_)), mlp_model.loss_curve_, color='purple', alpha=0.1) # Altını hafif boyama
plt.tight_layout()
plt.show()

# --- 3. NAIVE BAYES: Olasılık Dağılımı (Gaussian Distribution) ---
# Naive Bayes'in veriyi nasıl ayırdığını göstermek için en belirgin 2 özelliği seçiyoruz
features_to_plot = ['radius_mean', 'texture_mean']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Naive Bayes Olasılık Dağılımı (Çan Eğrileri)', fontsize=16, fontweight='bold')

for i, feature in enumerate(features_to_plot):
    # İyi huylu (0) ve Kötü huylu (1) hastalar için KDE (Kernel Density Estimation) eğrilerini çiziyoruz
    sns.kdeplot(data=df, x=feature, hue='diagnosis', fill=True, 
                palette={0: 'blue', 1: 'red'}, ax=axes[i], alpha=0.5)
    
    axes[i].set_title(f"'{feature}' Özelliğinin Dağılımı", fontsize=13)
    axes[i].set_xlabel(feature, fontsize=11)
    axes[i].set_ylabel('Olasılık Yoğunluğu', fontsize=11)
    
    # Lejantı düzeltme
    axes[i].legend(title='Teşhis', labels=['Kötü Huylu (1 - Kırmızı)', 'İyi Huylu (0 - Mavi)'])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()