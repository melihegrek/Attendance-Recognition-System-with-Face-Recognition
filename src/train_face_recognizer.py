import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Yüz algılama için MTCNN modelini yükleyelim
mtcnn = MTCNN(keep_all=True)

# Yüz tanıma için FaceNet modelini yükleyelim
model = InceptionResnetV1(pretrained='vggface2').eval()

# Veri seti yolu ve kişi isimlerini al
dataset_path = 'lfw-deepfunneled'  # LFW veri seti dizini
person_names = os.listdir(dataset_path)  # Klasörlerdeki kişi isimlerini al

# Yüz özelliklerini saklamak için bir liste
faces = []
labels = []

# Veri setindeki her kişiyi işleyelim
for label, person in enumerate(person_names):
    person_path = os.path.join(dataset_path, person)  # Her bir kişinin klasörüne gir

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)  # Fotoğrafın tam yolunu al

        # Fotoğrafı oku ve boyutlandır
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatı, FaceNet RGB formatı bekliyor

        # Yüzleri tespit et
        faces_detected = mtcnn(img_rgb)

        # Yüzler varsa
        if faces_detected is not None:
            for face in faces_detected:
                # Yüzün embedding'ini çıkar
                embedding = model(face.unsqueeze(0)).detach().cpu().numpy()
                faces.append(embedding)
                labels.append(label)  # Etiket olarak kişiyi ekle

# Modeli kaydetmek için pickle veya numpy formatını kullanabiliriz
np.save('faces.npy', faces)  # Yüz özelliklerini kaydediyoruz
np.save('labels.npy', labels)  # Etiketleri kaydediyoruz
print("Eğitim tamamlandı ve model kaydedildi!")
