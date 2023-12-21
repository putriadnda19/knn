# app.py

import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fungsi untuk melatih model KNN
def train_knn_model(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Fungsi untuk mengevaluasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    st.title("Aplikasi KNN dengan Streamlit")

    # Tambahkan elemen-elemen UI
    st.sidebar.header("Pengaturan Model")
    
    k_value = st.sidebar.slider("Jumlah Tetangga (k)", 1, 10, 3)
    
    AGE = st.number_input('Input umur pasien')

    SMOKING = ['YES', 'NO']
    SMOKING = st.radio('Apakah pasien merokok?', SMOKING)
    
    YELLOW_FINGERS = ['YES', 'NO']
    YELLOW_FINGERS = st.radio('Apakah pasien jari pasien kuning?', YELLOW_FINGERS)
    
    AXIENTY = ['YES', 'NO']
    AXIENTY = st.radio('Apakah pasien mempunyai kecemasan berlebih?', AXIENTY)
    
    PEER_PRESSURE = ['YES', 'NO']
    PEER_PRESSURE= st.radio('Apakah pasien mempunyai tekanan dari teman sebaya?', PEER_PRESSURE)
    
    COUGHING = ['YES', 'NO']
    COUGHING = st.radio('Apakah pasien batuk-batuk?', COUGHING)
    
    SHORTNESS_OF_BREATH = ['YES', 'NO']
    SHORTNESS_OF_BREATH = st.radio('Apakah pasien sesak nafas?', SHORTNESS_OF_BREATH)
    
    SWALLOWING_DIFFICULTY = ['YES', 'NO']
    SWALLOWING_DIFFICULTY = st.radio('Apakah pasien kesulitan menelan?', SWALLOWING_DIFFICULTY)
    
    CHEST_PAIN = ['YES', 'NO']
    CHEST_PAIN = st.radio('Apakah pasien nyeri dada?', CHEST_PAIN)
    
    CHRONIC_DISEASE = ['YES', 'NO']
    CHRONIC_DISEASE = st.radio('Apakah pasien mempunyai penyakit kronis?', CHRONIC_DISEASE)
    
    WHEEZING = ['YES', 'NO']
    WHEEZING = st.radio('Apakah pasien mengi (Napas Berbunyi)?', WHEEZING)

    # Muat dataset (ganti dengan dataset Anda)
    # Misalnya, Anda dapat menggunakan dataset iris untuk contoh
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Pisahkan data menjadi data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    model = train_knn_model(X_train, y_train, n_neighbors=k_value)

    # Evaluasi model
    accuracy = evaluate_model(model, X_test, y_test)

    # Tampilkan hasil evaluasi
    st.sidebar.subheader("Hasil Evaluasi Model")
    st.sidebar.write(f"Akurasi: {accuracy:.2%}")

if __name__ == "__main__":
    main()
