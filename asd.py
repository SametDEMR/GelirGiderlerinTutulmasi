import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

class SesTanimaModeli:
    def __init__(self, model_path="SVM_linear_model.pkl", dataset_path="veriseti.csv", threshold=0.6):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.threshold = threshold

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, duration=15)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        return mfccs

    def load_or_train_model(self, X, y):
        if not os.path.exists(self.model_path):
            print("Model dosyası bulunamadı, yeni bir model eğitiliyor...")
            model = SVC(kernel='linear', probability=False, C=1.0, random_state=42)
            model.fit(X, y)
            joblib.dump(model, self.model_path)
            print("Yeni model kaydedildi.")
        else:
            print("Model dosyası bulundu. Yükleniyor...")
            model = joblib.load(self.model_path)
        return model

    def create_dataset(self, data_dirs):
        features = []
        labels = []
        for directory in data_dirs:
            folder_path = os.path.join(os.getcwd(), directory)
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    label = directory
                    features.append(self.extract_features(file_path))
                    labels.append(label)
        X = np.array(features)
        y = np.array(labels)
        return X, y

    def save_dataset_to_csv(self, X, y):
        df = pd.DataFrame(X)
        df['label'] = y
        df.to_csv(self.dataset_path, index=False)
        print(f"Veri seti CSV dosyası olarak kaydedildi: {self.dataset_path}")

    def predict(self, file_path):
        try:
            model = joblib.load(self.model_path)
            features = self.extract_features(file_path).reshape(1, -1)
            prediction = model.predict(features)[0]

            # Tahminin tanınma eşik kontrolü (sınır belirleme)
            distances = model.decision_function(features)
            confidence = np.max(np.abs(distances))  # Güven skoru

            if confidence < self.threshold:
                print("Yeni bir ses tespit edildi. Eğitim veri setine eklenebilir.")
                return "tanınmayan", features
            else:
                print(f"Bu ses tanındı: {prediction} (Güven: {confidence:.2f})")
                return prediction, None
        except Exception as e:
            print(f"Hata oluştu: {e}")
            return None, None

    def add_to_training_data(self, features, label):
        try:
            if not os.path.exists(self.dataset_path):
                print("Veri seti dosyası bulunamadı. Yeni veri seti oluşturuluyor...")
                df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
                df['label'] = label
                df.to_csv(self.dataset_path, index=False)
                print("Yeni veri seti başarıyla oluşturuldu.")
            else:
                df = pd.read_csv(self.dataset_path)
                new_data = pd.DataFrame([features.flatten().tolist() + [label]], columns=df.columns)
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(self.dataset_path, index=False)
                print("Yeni veri başarıyla veri setine eklendi.")
        except Exception as e:
            print(f"Hata oluştu: {e}")

    def retrain_model(self):
        try:
            df = pd.read_csv(self.dataset_path)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = SVC(kernel='linear', probability=False, C=1.0, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, self.model_path)
            print("Model başarıyla yeniden eğitildi ve kaydedildi.")
        except Exception as e:
            print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    ses_tanima = SesTanimaModeli()

    data_dirs = ["veli", "guray", "salih"]
    X, y = ses_tanima.create_dataset(data_dirs)
    ses_tanima.save_dataset_to_csv(X, y)
    model = ses_tanima.load_or_train_model(X, y)

    file_path = "SesTanima/samet.wav"
    prediction, features = ses_tanima.predict(file_path)

    if prediction == "tanınmayan" and features is not None:
        label = input("Bu ses için bir etiket girin: ")
        ses_tanima.add_to_training_data(features, label)
        ses_tanima.retrain_model()
