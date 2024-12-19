---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

IMPORT LIBRARY
``` python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
```

LOAD DATA IRIS
``` python
bungaIris = load_iris()
X, Y = bungaIris.data, bungaIris.target
```


MEMISAHKAN DATA PRLATIHAN & DATA UJI
``` python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

MODEL k-Nearest Neighbors (k-NN),
``` python
k = 3
model = KNeighborsClassifier(n_neighbors=k)

MELATIH DATA
model.fit(X_train, Y_train)
```

MEMPREDIKSI DATA UJI 
``` python
Y_prediksi = model.predict(X_test)
```
EVALUASI MODEL
``` python
Akurasi = accuracy_score(Y_test, Y_prediksi)
print(f"Akurasi model: {Akurasi * 100:.2f}%")
print("\nLaporan Klasifikasi:\n")
print(classification_report(Y_test, Y_prediksi, target_names=bungaIris.target_names))
```

::: OUTPUT = Akurasi model: 100.00%

    Laporan Klasifikasi:

                  precision    recall  f1-score   support

          setosa       1.00      1.00      1.00        10
      versicolor       1.00      1.00      1.00         9
       virginica       1.00      1.00      1.00        11

        accuracy                           1.00        30
       macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30

TESTING MENGGUNAKAN DATA BARU
``` python
data_baru = [[5.1, 3.5, 1.4, 0.2]]  # Data bunga baru
prediksi = model.predict(data_baru)
nama_kelas = bungaIris.target_names[prediksi[0]]
print(f"Data baru diklasifikasikan sebagai: {nama_kelas}")
```

::: OUTPUT = Data baru diklasifikasikan sebagai: setosa