import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Firebase Firestore ile etkileşimde bulunmak için gereken yardımcı fonksiyonlar.

# Firebase'e bağlanmak için kimlik bilgilerini ayarlayın
cred = credentials.Certificate('C:/Users/Huseyin/Desktop/App/credentials.json')
firebase_admin.initialize_app(cred)

# Firestore istemcisini oluşturun
db = firestore.client()

def get_data_from_firestore():
    
    """
    Firestore'dan tüm kullanıcı verilerini alır ve bir pandas DataFrame olarak döner.
    """
    users_ref = db.collection('Users')
    docs = users_ref.stream()

    data = []
    for doc in docs:
        data.append(doc.to_dict())

    df = pd.DataFrame(data)
    return df

def save_user_selection(user_id, mood, suggestion):
    """
    Kullanıcının belirli bir duygusal durum için yaptığı aktivite seçimini Firestore'a kaydeder.
    """
    users_ref = db.collection('Users')
    user_ref = users_ref.document(user_id)
    
    field = f'{mood.capitalize()}kenSelections'
    
    user_ref.set({
        field: {suggestion: firestore.Increment(1)}
    }, merge=True)

def get_most_frequent_activity(user_id, mood):
    
    """
    Belirli bir kullanıcı ve duygusal durum için en çok tercih edilen aktiviteyi döner.
    """
    users_ref = db.collection('Users')
    user_ref = users_ref.document(user_id)
    user_data = user_ref.get().to_dict()
    
    field = f'{mood.capitalize()}kenSelections'
    
    if field not in user_data:
        return None

    activities = user_data[field]
    most_frequent_activity = max(activities, key=activities.get)
    
    return most_frequent_activity

def get_most_frequent_activity_all_users(mood):
    
    """
    Tüm kullanıcıların belirli bir duygusal durum için en çok tercih ettiği aktiviteyi döner.
    """
    users_ref = db.collection('Users')
    docs = users_ref.stream()

    activity_counts = {}

    for doc in docs:
        user_data = doc.to_dict()
        field = f'{mood.capitalize()}kenSelections'
        if field in user_data:
            activities = user_data[field]
            for activity, count in activities.items():
                if activity not in activity_counts:
                    activity_counts[activity] = 0
                activity_counts[activity] += count

    if not activity_counts:
        return None

    most_frequent_activity = max(activity_counts, key=activity_counts.get)
    return most_frequent_activity