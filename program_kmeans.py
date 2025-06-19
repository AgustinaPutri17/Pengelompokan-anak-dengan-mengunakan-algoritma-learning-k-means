import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_path = r"C:\Users\agust\Downloads\AL\databayi_dataset.csv"
print("📁 Working directory:", os.getcwd())
print("📄 Cek keberadaan file:", os.path.exists(file_path))

if not os.path.exists(file_path):
    print("❌ File tidak ditemukan. Pastikan 'databayi_dataset.csv' ada di folder yang sama.")
    exit()
try:
    data = pd.read_csv(file_path)
    print("✅ Dataset berhasil dibaca!\n")
    print(data.head())
except Exception as e:
    print(f"❌ Gagal membaca file CSV: {e}")
    exit()

if 'Status Gizi' not in data.columns:
    print("❌ Kolom 'Status Gizi' tidak ditemukan.")
    exit()

data = data[data['Status Gizi'].str.lower() == 'normal']
print(f"✅ Jumlah data anak dengan Status Gizi 'Normal': {len(data)}")
if data.empty:
    print("⚠️ Tidak ada data anak dengan Status Gizi 'Normal' dalam dataset.")
    exit()
label_encoder = LabelEncoder()
if 'Jenis Kelamin' in data.columns:
    data['Jenis Kelamin'] = label_encoder.fit_transform(data['Jenis Kelamin'])  # L=1, P=0
else:
    print("❌ Kolom 'Jenis Kelamin' tidak ditemukan.")
    exit()

required_columns = ['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']
if not all(col in data.columns for col in required_columns):
    print("❌ Dataset tidak memiliki semua kolom yang dibutuhkan:", required_columns)
    exit()

X = data[required_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_

print("\n📊 Hasil Clustering pada Anak dengan Status Gizi Normal:")
print(data[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)', 'Cluster']])

plt.figure(figsize=(10, 7))

colors = ['pink', 'blue']
labels = ['Cluster 0', 'Cluster 1']

for cluster in range(kmeans.n_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(
        cluster_data['Tinggi Badan (cm)'],
        cluster_data['Umur (bulan)'],
        s=100,
        c=colors[cluster],
        label=labels[cluster],
        alpha=0.6,
        edgecolor='k'
    )
centroids = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids)  
plt.scatter(
    centroids_original[:, 2],  
    centroids_original[:, 0],  
    s=200,
    c='red',
    label='Centroid',
    marker='X'
)
plt.xlabel("Tinggi Badan (cm)", fontsize=12)
plt.ylabel("Umur (bulan)", fontsize=12)
plt.title("Visualisasi Cluster Anak dengan Status Gizi Normal", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

