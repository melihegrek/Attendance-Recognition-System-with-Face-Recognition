import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Yüz algılama için MTCNN modelini yükleyelim
mtcnn = MTCNN(keep_all=True)

# Yüz tanıma için FaceNet modelini yükleyelim
model = InceptionResnetV1(pretrained='vggface2').eval()

# Eğitilmiş yüz özelliklerini ve etiketlerini yükle
faces = np.load('custom_faces.npy', allow_pickle=True)
labels = np.load('custom_labels.npy', allow_pickle=True)

# Veri seti yolu ve kişi isimlerini al
dataset_path = 'custom_train_img'  # Veri seti dizini
person_names = os.listdir(dataset_path)  # Klasörlerdeki kişi isimlerini al

# Web kamera açalım
cap = cv2.VideoCapture(0)  # 0, yerel kamerayı temsil eder

while True:
    # Kameradan görüntü alalım
    ret, frame = cap.read()
    
    if not ret:
        print("Camera does not open!")
        break
    
    # Görüntüyü RGB formatına çevirelim , OpenCV by default captures images in BGR format
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Yüzleri algılayalım
    boxes, _ = mtcnn.detect(img_rgb)
    faces_in_frame = mtcnn(img_rgb)

    # Eğer yüzler tespit edildiyse
    if faces_in_frame is not None:
        for i, box in enumerate(boxes):
            # Yüzün etrafına dikdörtgen çizelim
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil renkli kutu

            # Yüzün embedding'ini çıkaralım
            face = faces_in_frame[i]
            test_embedding = model(face.unsqueeze(0)).detach().cpu().numpy()

            # En yakın eşleşmeyi bulmak için mesafeleri karşılaştıralım
            min_distance = float('inf')
            best_match = None
            
            for i, stored_embedding in enumerate(faces):
                distance = np.linalg.norm(test_embedding - stored_embedding)  # Euclidean mesafesini hesapla
                if distance < min_distance:
                    min_distance = distance
                    best_match = labels[i]  # En yakın yüzün etiketini al

            # Doğruluk oranını hesaplayalım
            max_distance = 1.0  # Kabul edilebilir maksimum mesafe
            similarity_score = max(0, (1 - min_distance / max_distance)) * 100  # Normalize edilmiş doğruluk yüzdesi
            
            # Sonuçları ekranda gösterelim
            if best_match is not None and min_distance <= max_distance:
                name = person_names[best_match]
                text = f"{name}: {similarity_score:.2f}%"
                # Yüz kutusunun üst kısmına ismi yazdıralım
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                text = "Unknown"
                # Yüz kutusunun üst kısmına "Tanınmadı" yazalım
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Görüntüyü ekranda gösterelim
    cv2.imshow("Canlı Yüz Tanıma", frame)

    # 'q' tuşuna basılınca döngüden çıkalım
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakalım ve pencereleri kapatalım
cap.release()
cv2.destroyAllWindows()
