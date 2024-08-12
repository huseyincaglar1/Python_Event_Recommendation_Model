from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from firebase_utils import get_data_from_firestore, save_user_selection, get_most_frequent_activity, get_most_frequent_activity_all_users
from model_utils import update_and_train_model

app = Flask(__name__)

# Model ve Label Encoder dosyalarının tam yolunu belirleyin
model_dir = 'C:/Users/Huseyin/Desktop/App'
modelHappy = joblib.load(os.path.join(model_dir, 'rfHappyModel.pkl'))
modelSad = joblib.load(os.path.join(model_dir, 'rfSadModel.pkl'))
modelAngry = joblib.load(os.path.join(model_dir, 'rfAngryModel.pkl'))
le_gender = joblib.load(os.path.join(model_dir, 'le_gender.pkl'))
le_job = joblib.load(os.path.join(model_dir, 'le_job.pkl'))
le_happy = joblib.load(os.path.join(model_dir, 'le_happy.pkl'))
le_sad = joblib.load(os.path.join(model_dir, 'le_sad.pkl'))
le_angry = joblib.load(os.path.join(model_dir, 'le_angry.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    
    """
    Belirli bir kullanıcı ve duygusal durum için aktivite tahmini yapar.
    """
    data = request.json
    yas = data['yas']
    meslek = data['meslek']
    cinsiyet = data['cinsiyet']
    mood = data['mood'].lower()

    # Yeni kullanıcı verisini encode et
    try:
        encoded_gender = le_gender.transform([cinsiyet])[0]
    except ValueError:
        return jsonify({'error': 'Bilinmeyen cinsiyet girildi, lütfen geçerli bir cinsiyet girin.'})

    try:
        encoded_job = le_job.transform([meslek])[0]
    except ValueError:
        encoded_job = le_job.transform(['other'])[0]

    user_features = pd.DataFrame([[yas, encoded_gender, encoded_job]], columns=['Yaş', 'Cinsiyet', 'Meslek'])

    # Seçilen modeli yükleyin
    if mood == 'mutlu':
        model = modelHappy
        label_encoder = le_happy
    elif mood == 'üzgün':
        model = modelSad
        label_encoder = le_sad
    elif mood == 'öfkeli':
        model = modelAngry
        label_encoder = le_angry
    else:
        return jsonify({'error': 'Geçersiz mood seçimi. Lütfen "mutlu", "üzgün" veya "öfkeli" olarak girin.'})

    # Model ile tahmin yap
    probabilities = model.predict_proba(user_features)

    # En yüksek üç olasılığı seç
    top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
    top_3_activities = label_encoder.inverse_transform(top_3_indices)

    return jsonify({
        'predictions': top_3_activities.tolist()
    })

@app.route('/save_selection', methods=['POST'])
def save_selection():
    
    """
    Kullanıcı seçimini kaydeder ve modeli yeniden eğitir.
    """
    data = request.json
    user_id = data['user_id']
    mood = data['mood'].lower()
    suggestion = data['suggestion']
    
    # Kullanıcı önerisini Firebase'e kaydet
    try:
        save_user_selection(user_id, mood, suggestion)
        # Yeni kullanıcı tercihleriyle modeli yeniden eğitin
        df = get_data_from_firestore()

        # Kullanıcının en çok tercih ettiği aktiviteleri al
        user_most_frequent_happy = get_most_frequent_activity(user_id, 'mutlu')
        user_most_frequent_sad = get_most_frequent_activity(user_id, 'üzgün')
        user_most_frequent_angry = get_most_frequent_activity(user_id, 'öfkeli')

        # Eksik olan durumları genel kullanıcı verileriyle doldur
        if not user_most_frequent_happy:
            user_most_frequent_happy = get_most_frequent_activity_all_users('mutlu')
        if not user_most_frequent_sad:
            user_most_frequent_sad = get_most_frequent_activity_all_users('üzgün')
        if not user_most_frequent_angry:
            user_most_frequent_angry = get_most_frequent_activity_all_users('öfkeli')

        # Yeni veri seti oluştur
        new_data = {
            'Yaş': df['Age'].values[0],
            'Cinsiyet': df['Gender'].values[0],
            'Meslek': df['Occupation'].values[0],
            'Mutluyken': user_most_frequent_happy,
            'Üzgünken': user_most_frequent_sad,
            'Öfkeliyken': user_most_frequent_angry
        }
        
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        update_and_train_model(df)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify({"message": "Selection successfully saved and model retrained"}), 200

@app.route('/get_most_frequent_activity', methods=['POST'])
def get_most_frequent_activity_endpoint():
    
    """
    Kullanıcıya özel ve genel olarak en çok tercih edilen aktiviteyi döner.
    """
    data = request.json
    user_id = data.get('user_id')
    mood = data.get('mood', '').lower()

    if user_id == 'all_users':
        most_frequent_activity_all = get_most_frequent_activity_all_users(mood)
        response = {
            "most_frequent_activity_all": most_frequent_activity_all
        }
    else:
        most_frequent_activity_user = get_most_frequent_activity(user_id, mood)
        most_frequent_activity_all = get_most_frequent_activity_all_users(mood)
        response = {
            "most_frequent_activity_user": most_frequent_activity_user,
            "most_frequent_activity_all": most_frequent_activity_all
        }

    return jsonify(response), 200

@app.route('/train_model', methods=['POST'])
def train_model():
    
    """
    Firebase'den verileri çekerek modeli yeniden eğitir.
    """
    # Firebase'den verileri çekin
    df = get_data_from_firestore()

    # Modeli yeniden eğitin
    update_and_train_model(df)

    return jsonify({"message": "Model successfully trained"}), 200

@app.route('/test')
def test():
    return jsonify({
        'test': 12345
    })

if __name__ == '__main__':
    app.run(debug=True, host="192.168.1.104")