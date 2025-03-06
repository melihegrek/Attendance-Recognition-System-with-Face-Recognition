#### Proje Tanımı:
Bu proje, yüz tanıma teknolojisini kullanarak bir katılım takip sistemi geliştirmeyi amaçlamaktadır. Sistem, bir web kamerası aracılığıyla canlı görüntü alır, yüzleri tanır ve daha önce eğitilmiş yüz özelliklerine dayanarak katılımı kaydeder. Katılım bilgileri bir CSV dosyasına kaydedilir, her kişi tanındığında isim ve zaman damgası ile bir kayıt yapılır.

---

#### Kullanılan Teknolojiler:
- **Python**: Proje dili
- **OpenCV**: Görüntü işleme için
- **MTCNN**: Yüz algılama için
- **FaceNet (InceptionResnetV1)**: Yüz tanıma için
- **NumPy**: Veri işlemleri ve hesaplamalar için
- **CSV**: Katılım verilerinin saklanması için
- **datetime**: Zaman damgası eklemek için

---

#### Proje Özellikleri:
- **Yüz Tanıma**: MTCNN kullanılarak kameradan alınan görüntülerdeki yüzler tespit edilir.
- **Eğitimli Yüz Tanıma**: Önceden eğitilmiş yüz embedding'leri (özellikleri) ile tanıma yapılır ve tanınan kişi belirlenir.
- **Katılım Takibi**: Tanınan kişinin ismi ve zaman damgası ile katılım verileri bir CSV dosyasına kaydedilir.
- **Kişi Ekleme**: Tanınan kişi daha önce eklenmediyse ve doğruluk oranı %25'ten fazla ise katılım kaydedilir.
- **Gerçek Zamanlı Tanıma**: Gerçek zamanlı video akışıyla yüz tanıma yapılır ve tanınan kişinin bilgileri ekranda görüntülenir.

---

#### Proje Akışı:
1. **Yüz Algılama ve Tanıma:**
   - İlk olarak **MTCNN** (Multi-task Cascaded Convolutional Networks) kullanılarak yüzler tespit edilir.
   - Eğer yüzler tespit edilirse, **FaceNet** (InceptionResnetV1) modeli kullanılarak her bir yüzün özellikleri çıkarılır ve bu özellikler, daha önce eğitilmiş verilerle karşılaştırılır.
   
2. **Yüz Özellikleri Karşılaştırma:**
   - Her bir yüzün özellikleri (embedding) çıkarıldıktan sonra, **Euclidean mesafesi** kullanılarak tanınan yüzün daha önce kaydedilen yüzlerle olan benzerliği hesaplanır.
   - Eğer mesafe belirli bir eşik değerinden (örneğin, 1.0) küçükse, o kişi tanınmış olur.

3. **Katılım Kaydı:**
   - Tanınan kişinin adı, doğruluk oranıyla birlikte ekrana yazdırılır ve **CSV dosyasına** katılım bilgisi eklenir.
   - Daha önce kaydedilmeyen bir kişi tanındığında, bu kişi **katılım listesine eklenir** ve zaman damgası ile kaydedilir.

---

#### Kullanılan Kütüphaneler:
1. **OpenCV**: Görüntü işleme ve video akışı için kullanılır.
2. **facenet-pytorch**: Yüz tanıma ve yüz embedding'leri için kullanılır.
3. **NumPy**: Matematiksel hesaplamalar için kullanılır, özellikle yüz karşılaştırmalarında.
4. **CSV**: Katılım bilgilerini saklamak için kullanılır.

---

#### Ana Kod Açıklaması:
Bu proje, iki ana bileşenden oluşur: 
1. **MTCNN** kullanarak yüz algılama.
2. **FaceNet** (InceptionResnetV1) kullanarak yüz tanıma.

**Kodun Adımları:**

1. **Model Yükleme:**
   - MTCNN, yüz tespiti için kullanılır.
   - FaceNet modelini **pretrained='vggface2'** parametresiyle yükleriz. Bu model, yüzlerin özelliklerini çıkarabilen bir derin öğrenme modelidir.

2. **CSV Dosyasının Hazırlanması:**
   - Katılım verileri, **17_December_attendance_list.csv** dosyasına yazılır. Başlangıçta bu dosya her çalıştırmada sıfırlanır ve başlık eklenir.
   
3. **Kamera Akışı:**
   - **cv2.VideoCapture(0)** ile yerel kamera başlatılır ve sürekli olarak görüntü alınır.
   - Alınan her görüntü, **MTCNN** ile analiz edilir ve yüzler tespit edilir.
   
4. **Yüz Tanıma ve Katılım Kaydı:**
   - Tespit edilen her yüz için **FaceNet** ile embedding çıkarılır. Bu embedding'ler, önceden kaydedilen **embedding'lerle** karşılaştırılır.
   - Eğer mesafe belirli bir eşiğin altındaysa, kişinin adı kaydedilir ve katılım listesine eklenir.
   
5. **Gerçek Zamanlı Sonuç Gösterimi:**
   - Yüzün etrafına dikdörtgen çizilir ve kişinin adı ve doğruluk oranı ekranda gösterilir.
   - Eğer kişi tanınmazsa, ekranda "Unknown" yazısı çıkar.

6. **CSV Dosyasına Yazma:**
   - Tanınan kişi daha önce kaydedilmemişse ve doğruluk oranı %25'in üzerindeyse, **CSV dosyasına** o kişinin adı ve zaman damgası kaydedilir.

---

#### Gereksinimler:
1. Python 3.x
2. Aşağıdaki kütüphaneler:
    - OpenCV: `pip install opencv-python`
    - NumPy: `pip install numpy`
    - facenet-pytorch: `pip install facenet-pytorch`
3. Yüz tanıma için önceden eğitilmiş model dosyaları:
    - `custom_faces.npy`: Yüz özelliklerini içeren dosya
    - `custom_labels.npy`: Kişi etiketlerini içeren dosya

---

#### Kurulum:
1. Bu repository'yi indirin.
2. İhtiyaç duyulan Python kütüphanelerini yükleyin:
3. Kamera ile yüz tanıma işlemini başlatın:
    ```bash
    python face_recognition_attendance.py
    ```

---

#### Kullanım:
1. Program başladığında, kamera görüntüsü ekranda belirecektir.
2. Yüzler algılandığında, tanınan kişilerin isimleri ve doğruluk yüzdeleri ekranda görünecektir.
3. Eğer bir kişi tanınmışsa ve doğruluk oranı %25'in üzerinde ise, katılım kaydına isim ve zaman damgası eklenir.
4. Katılım bilgileri, `17_December_attendance_list.csv` dosyasına kaydedilir.
5. Programdan çıkmak için 'q' tuşuna basın.

---

#### Katılım CSV Formatı:
CSV dosyasında her katılım kaydının şu formatta olması beklenir:
- **Name**: Kişinin ismi
- **Timestamp**: Kişinin tanındığı tarih ve saat

Örnek:
```csv
Name, Timestamp
John Doe, 2025-12-17 10:30:00
Jane Smith, 2025-12-17 10:32:15
```

---

#### Uyarılar:
- Yüz tanıma modelinin doğru çalışabilmesi için, yüzlerin doğrudan kameraya bakması ve yeterli aydınlatmaya sahip bir ortamda bulunması önemlidir.
- Model ve etiket dosyalarının (`custom_faces.npy` ve `custom_labels.npy`) doğru şekilde yüklenmesi gerekmektedir.

---

#### İleri Düzey Özellikler:
- **Yeni Kişi Ekleme**: Yeni yüzler eklenebilir ve model güncellenebilir.
- **Farklı Modeller**: Yüz tanıma için farklı model seçenekleri kullanılabilir.
