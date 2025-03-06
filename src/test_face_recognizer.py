import cv2
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

# Yüz algılama için MTCNN modelini yükleyelim
mtcnn = MTCNN(keep_all=True)

# Yüz tanıma için FaceNet modelini yükleyelim
model = InceptionResnetV1(pretrained='vggface2').eval()

# Eğitilmiş yüz özelliklerini ve etiketlerini yükle
faces = np.load('faces.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# Veri seti yolu ve kişi isimlerini al
dataset_path = 'lfw-deepfunneled'  # LFW veri seti dizini
person_names = os.listdir(dataset_path)  # Klasörlerdeki kişi isimlerini al

# Test fotoğrafını yükleyelim
test_image_path = "testimg9.jpg"  # Test fotoğrafının yolu
test_img = cv2.imread(test_image_path)
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Test fotoğrafındaki yüzleri algılayalım
boxes, _ = mtcnn.detect(test_img_rgb)  # Yüz kutularını (boxes) algılayalım
faces_in_test = mtcnn(test_img_rgb)

# Eğer test fotoğrafında yüz tespit edildiyse
if faces_in_test is not None:

    for i, box in enumerate(boxes):
        # Koordinatları tamsayıya çevir
        x1, y1, x2, y2 = [int(coord) for coord in box]

        # Yüzün etrafına dikdörtgen çizelim
        cv2.rectangle(test_img, 
                      (x1, y1), 
                      (x2, y2), 
                      (0, 255, 0), 2)  # Yeşil renkli kutu, kalınlık 2 px

        # Test fotoğrafındaki yüzün embedding'ini çıkaralım
        face = faces_in_test[i]
        test_embedding = model(face.unsqueeze(0)).detach().cpu().numpy()

        # En yakın eşleşmeyi bulmak için mesafeleri karşılaştıralım
        min_distance = float('inf')
        best_match = None

        for j, stored_embedding in enumerate(faces):
            distance = np.linalg.norm(test_embedding - stored_embedding)  # Euclidean mesafesini hesapla
            if distance < min_distance:
                min_distance = distance
                best_match = labels[j]  # En yakın yüzün etiketini al

        # Doğruluk oranını hesaplayalım (örnek eşik: max_distance = 1.0)
        max_distance = 1.0  # Kabul edilebilir maksimum mesafe
        similarity_score = max(0, (1 - min_distance / max_distance)) * 100  # Normalize edilmiş doğruluk yüzdesi

        # Sonuçları yazdıralım ve yüzün üzerine adını yazalım
        if best_match is not None and min_distance <= max_distance:
            name = person_names[best_match]
            print(f"Test fotoğrafı, kişi {name}'a ait. Mesafe: {min_distance:.2f}")
            print(f"Doğruluk oranı: {similarity_score:.2f}%")

            # İsmi kutunun üstüne yazalım
            cv2.putText(test_img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print("Test fotoğrafı tanınmadı.")
            print(f"Mesafe: {min_distance:.2f}. Doğruluk oranı çok düşük.")

            # Tanınmayan yüz için "Tanınmadı" yazalım
            cv2.putText(test_img, "Tanınmadı", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
else:
    print("Test fotoğrafında yüz bulunamadı.")

# Test fotoğrafını bir OpenCV penceresinde gösterelim
cv2.imshow("Detected and Recognized Faces", test_img)
cv2.waitKey(0)  # Tuşa basıldığında pencereyi kapat
cv2.destroyAllWindows()
