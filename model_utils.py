from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

model_dir = 'C:/Users/Huseyin/Desktop/App'

def fill_missing_with_averages(df):
    # Her bir duygusal durum için ortalama verileri hesapla
    average_happy = df['Mutluyken'].mode()[0]
    average_sad = df['Üzgünken'].mode()[0]
    average_angry = df['Öfkeliyken'].mode()[0]

    # Eksik değerleri genel kullanıcı ortalamaları ile doldur
    df['Mutluyken'].fillna(average_happy, inplace=True)
    df['Üzgünken'].fillna(average_sad, inplace=True)
    df['Öfkeliyken'].fillna(average_angry, inplace=True)
    
    return df

def update_and_train_model(df):
    # Eksik verileri ortalama verilerle doldur
    df = fill_missing_with_averages(df)
    
    # Kategorik değişkenleri kodlama
    label_encoders = {}
    for column in ['Cinsiyet', 'Meslek', 'Mutluyken', 'Üzgünken', 'Öfkeliyken']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Özellikleri ve etiketleri belirleme
    features = df[['Yaş', 'Cinsiyet', 'Meslek']]
    labels_happy = df['Mutluyken']
    labels_sad = df['Üzgünken']
    labels_angry = df['Öfkeliyken']

    def train_and_evaluate(features, labels, label_encoder, state):
        if labels.isnull().all():
            print(f"No data available for {state} state, skipping training.")
            return None, None

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        target_names = [str(label) for label in label_encoder.classes_]
        print(f"{state} State Classification Report (with target names):")
        print(classification_report(y_test, y_pred, target_names=target_names, labels=label_encoder.transform(label_encoder.classes_), zero_division=0))
        return model, label_encoder

    # Mutluyken, Üzgünken ve Öfkeliyken durumları için modelleri yeniden eğitme
    modelHappy, le_happy = train_and_evaluate(features, labels_happy, label_encoders['Mutluyken'], 'Happy')
    modelSad, le_sad = train_and_evaluate(features, labels_sad, label_encoders['Üzgünken'], 'Sad')
    modelAngry, le_angry = train_and_evaluate(features, labels_angry, label_encoders['Öfkeliyken'], 'Angry')

    # Modelleri ve kodlayıcıları sadece eğitilmiş olanları kaydetme
    if modelHappy:
        joblib.dump(modelHappy, os.path.join(model_dir, 'rfHappyModel.pkl'))
        joblib.dump(le_happy, os.path.join(model_dir, 'le_happy.pkl'))
    if modelSad:
        joblib.dump(modelSad, os.path.join(model_dir, 'rfSadModel.pkl'))
        joblib.dump(le_sad, os.path.join(model_dir, 'le_sad.pkl'))
    if modelAngry:
        joblib.dump(modelAngry, os.path.join(model_dir, 'rfAngryModel.pkl'))
        joblib.dump(le_angry, os.path.join(model_dir, 'le_angry.pkl'))
    
    joblib.dump(label_encoders['Cinsiyet'], os.path.join(model_dir, 'le_gender.pkl'))
    joblib.dump(label_encoders['Meslek'], os.path.join(model_dir, 'le_job.pkl'))