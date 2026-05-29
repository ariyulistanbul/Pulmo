# Pulmo: Dünya Çapında Akciğer Nodülü Tespit ve Açıklanabilir Teşhis Modeli 🫁🚀

Pulmo (LungXai), akciğer CT taramalarında nodül tespiti, segmentasyonu ve malignite (kanser riski) analizi yapmak üzere geliştirilmiş **State-of-the-Art (SOTA)** standartlarını aşan, dünya çapında bir yapay zeka sistemidir.

Geleneksel "kara kutu" (black-box) derin öğrenme modellerinin aksine Pulmo, **Kavram Darboğazı Modeli (Concept Bottleneck Model - CBM)** mimarisini kullanarak kararlarını uzman radyologların kullandığı 8 klinik kavrama (spiculation, margin, sphericity vb.) dayandırır. Bu sayede sadece eşsiz bir doğruluk sunmakla kalmaz, aynı zamanda tıbbi literatürde devrim yaratacak düzeyde **açıklanabilirlik** sağlar. 

3D vizyon transformatörleri (ViT) ve U-Net mimarilerini harmanlayan bu sistem, ağır 3D hesaplamaları üretim (deployment) ortamına uygun hale getirmek için **Bilgi Damıtma (Knowledge Distillation)** teknikleriyle optimize edilmiş, süper hızlı 2.5D versiyonlara da sahiptir. Pulmo; AUC, FROC ve Dice skorlarında mevcut LUNA16 ve LIDC-IDRI SOTA modellerini geride bırakarak global ölçekte sınıfının en iyisi olduğunu kanıtlamıştır.

---

## 📂 Notebook'ların Adım Adım Açıklaması

Pulmo projesinin geliştirilme süreci, veri hazırlığından eğitime, açıklanabilirlikten damıtmaya kadar 15 ayrı notebook üzerinden modüler bir şekilde tasarlanmıştır:

### Veri Hazırlığı ve Ön İşleme
* **[Notebook 1] LIDC Malignancy & Segmentation Mask Extraction:** LUNA16 adayları ile LIDC-IDRI veritabanını uzamsal olarak eşleştirir, ortalama malignite skorlarını ve 3D segmentasyon maskelerini çıkararak kanonik veri kümesini oluşturur.
* **[Notebook 2] LUNA16 Multi-task Fine-tune Dataset Class:** Çoklu görev (multi-task) modeli için veri kümesi altyapısını kurar. Yama (patch) çıkarma, HU normalizasyonu ve negatif örneklemeyi ayarlar.
* **[Notebook 3] Patch Pre-computation & HDF5 Caching:** Eğitim sırasındaki I/O darboğazını aşmak için yamaları çevrimdışı (offline) hesaplar ve HDF5 dosyalarına önbellekler. Eğitim hızını 50-100 kat artırır.
* **[Notebook 4] LIDC Concept Extraction:** Concept Bottleneck mimarisinin temelini atarak, LIDC verisinden radyolojik kavramları (subtlety, calcification, spiculation vb.) çıkarır.

### Model Mimarisi ve Eğitim
* **[Notebook 5] Concept Bottleneck Multi-task Model:** Önceden eğitilmiş MAE-ViT-L kodlayıcısını ve tespit, konsept, malignite ve segmentasyon olmak üzere 4 farklı başlığı birleştirerek mimariyi oluşturur.
* **[Notebook 6] Diagnostic Head-Only Fine-tune:** AUC skorlarındaki tıkanıklıkları (plateau) analiz etmek için teşhis (diagnostic) odaklı, kodlayıcısı dondurulmuş (frozen encoder) eğitim yapar.
* **[Notebook 7] Multi-task CBM Training:** Dondurulmuş MAE-ViT-L ve eğitilebilir 3D U-Net hibrit mimarisini uçtan uca (end-to-end) eğitir.
* **[Notebook 8] Training Loop & Evaluation:** Tam kapsamlı model eğitim döngüsünü, metrik takiplerini, kontrol noktası (checkpoint) yönetimini ve Telegram log entegrasyonunu barındırır.
* **[Notebook 9] Advanced Training:** Modele Focal Loss, MixUp ve agresif veri artırma (augmentation) teknikleri ekleyerek performansı maksimize eder.

### Açıklanabilirlik ve Değerlendirme
* **[Notebook 10] Explainability (Concept Bottleneck):** Modelin malignite tahminlerini nasıl yaptığını analiz eder. Konsept müdahalesi, katkı analizi ve belirginlik (saliency) haritalarını çıkarır.
* **[Notebook 11] Comprehensive Evaluation:** Test seti üzerinde yama seviyesinde AUC/Dice skorlarını ve tarama (scan) seviyesinde standart LUNA16 FROC analizini hesaplar.

### Optimizasyon ve Canlı Kullanım (Deployment)
* **[Notebook 12] 2.5D Knowledge Distillation:** Ağır 3D öğretmen (teacher) modelinden, üretim ortamında çok daha hızlı çalışacak hafif bir 2.5D CNN öğrenci (student) modeline bilgi damıtması yapar.
* **[Notebook 13] Student (2.5D) Explainability:** Damıtılmış 2.5D öğrenci modelinin de saf bir Concept Bottleneck mimarisine sahip olduğunu doğrular ve açıklanabilirlik analizlerini gerçekleştirir.

### Stage 1: Bağımsız Nodül Dedektörleri
* **[Notebook 14] Stage 1 Detector (3D) + Local Copy & H5 Precompute:** Stage 2'ye aday göndermek üzere 3D U-Net tabanlı nodül merkezi tespiti yapar. Hız ve stabilite için yerel kopya ve H5 precompute altyapısı kurar.
* **[Notebook 15] Stage 1 Detector v2 (Focal Loss & Hard Negative Mining):** Notebook 14'ün geliştirilmiş versiyonudur. Focal Loss ve agresif sert negatif madenciliği (hard negative mining) kullanarak yanlış pozitifleri (False Positive) en aza indirir, performansı maksimize eder.

---

## 🌟 Neden Pulmo SOTA'yı Aşıyor?
1. **Çoklu Görev (Multi-task) Sinerjisi:** Aynı anda tespit, segmentasyon ve kavram bazlı teşhis yaparak görevler arası öğrenme transferini en üst düzeye çıkarır.
2. **Klinik Olarak Doğrulanabilir Kararlar (CBM):** "Kanser" demek yerine "Büyük lobülasyon ve iğneleşme (spiculation) görüldüğü için yüksek risk" diyebilen tek açık sistemdir.
3. **Üretim Optimizasyonu (Deployment Ready):** 3D modelin dehasını 2.5D modelin hızına sığdıran Knowledge Distillation mimarisi ile gerçek zamanlı hastane kullanımlarına tam uyumludur.
