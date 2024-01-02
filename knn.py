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
    st.title("Estimasi Pasien Yang Menderita Kanker Paru-Paru")

    # Tambahkan elemen-elemen UI
    st.sidebar.header("Navigasi")
    k_value = st.sidebar.slider("Jumlah Tetangga (k)", 1, 10, 3)
    st.write('***Selain input umur pasien kita hanya menginput 1 dan 2. 1=NO & 2=YES***')
    col1, col2 = st.columns(2)

    with col1 :
        AGE = st.slider('Input umur pasien', 1, 80)

    with col2 :
        SMOKING = st.number_input('Apakah pasien merokok?')

    with col1 :
        YELLOW_FINGERS = st.number_input('Apakah pasien jari pasien kuning?')

    with col2 :
        ANXIETY = st.number_input('Apakah pasien mempunyai kecemasan berlebih?')

    with col1 :
        PEER_PRESSURE = st.number_input('Apakah pasien mempunyai tekanan dari teman sebaya?')

    with col2 :
        COUGHING = st.number_input('Apakah pasien batuk-batuk?')

    with col1 :
        SHORTNESS_OF_BREATH = st.number_input('Apakah pasien sesak nafas?')

    with col2 :
        SWALLOWING_DIFFICULTY = st.number_input('Apakah pasien kesulitan menelan?')

    with col1 :
        CHEST_PAIN = st.number_input('Apakah pasien nyeri dada?')

    with col2 :
        CHRONIC_DISEASE = st.number_input('Apakah pasien mempunyai penyakit kronis?')

    with col1 :
        WHEEZING = st.number_input('Apakah pasien mengi (Napas Berbunyi)?')

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
