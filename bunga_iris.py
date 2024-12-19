from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

bungaIris = load_iris()
X, Y = bungaIris.data, bungaIris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

k = 3  # Jumlah tetangga terdekat
model = KNeighborsClassifier(n_neighbors=k)

# Latih model dengan data latih
model.fit(X_train, Y_train)

Y_prediksi = model.predict(X_test)

Akurasi = accuracy_score(Y_test, Y_prediksi)
print(f"Akurasi model: {Akurasi * 100:.2f}%")
print("\nLaporan Klasifikasi:\n")
print(classification_report(Y_test, Y_prediksi, target_names=bungaIris.target_names))

data_baru = [[5.1, 3.5, 1.4, 0.2]]  # Data bunga baru
prediksi = model.predict(data_baru)
nama_kelas = bungaIris.target_names[prediksi[0]]
print(f"Data baru diklasifikasikan sebagai: {nama_kelas}")