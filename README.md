# Perspektif Düzeltme

Bu proje, OpenCV kullanarak bir görüntüdeki belirli bir alanın örneğin bir belge veya dikdörtgen nesnenin perspektifini düzelten bir Python sınıfı içerir.

# Perspektif Dönüşüm
Bir kamera ya da gözlemci, üç boyutlu bir dünyayı iki boyutlu bir görüntüye dönüştürürken perspektif etkisi oluşur. Bu nedenle, nesneler uzaktan küçülür, yakın olanlar büyük görünür, paralel çizgiler görüntüde kesişiyormuş gibi görünür.
Perspektif dönüşüm ile, görüntüdeki bu perspektif bozulması düzeltilerek, örneğin bir eğik çekilmiş belge fotoğrafı, sanki doğrudan önünden çekilmiş gibi düzeltilir.

**Temel Mantık: **
- Görüntüdeki kenarları Canny ile tespit eder ve conturlarını oluşturur.
- Conturları büyükten küçüğe sıralar ve en büyük konturun köşelerini bulur
- Bulunan 4 köşe ise koşeleri perspektif fonksiyonunun anlayacağı bir şekilde sıralar.
- perspektif uygulanır kaydedilir ve ekrana yansıtılır.

## Avantajları
- Eğik çekilmiş görüntüleri gerçek ve düz görünüme çevirir.
- Belge fotoğraflarını okunabilir hale getirir.
- Nesnelerin ölçümlerini doğru yapmayı sağlar.
- Farklı açılardan alınan görüntüleri hizalayarak birleştirmeyi kolaylaştırır.

# Perspektif Düzeltme uygulaması

`main.py`
- Görüntüde kenarları ve en büyük konturu tespit eder.
- Bu konturun köşe noktalarını belirler ve sıralar.
- Köşe noktalarına göre perspektif dönüşümü uygulayarak görüntüyü düzleştirir.
- Düzeltmiş görüntüyü ekranda gösterir ve dosyaya kaydeder.

Aşağıda Python kodu ve açıklamaları yer almaktadır.

```python
import cv2
import numpy as np


class PerspectiveCorrection:
    def __init__(self, img):
        self.img = img  # İşlenecek orijinal görüntü
        self.tump = []  # Köşe noktalarını tutacak liste


    def edge_detection(self):
        # Görüntüyü gri tonlamaya çevir
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Gürültüyü azaltmak için Gaussian bulanıklık uygula
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Kenarları tespit etmek için Canny algoritması kullan
        edged = cv2.Canny(gray, 130, 250)
        # Kenarları kalınlaştırmak için dilate işlemi uygula
        edged = cv2.dilate(edged, None, iterations=1)
        # Kenarları biraz inceltmek için erode işlemi uygula
        edged = cv2.erode(edged, None, iterations=1)

        # Konturları bul
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Alanlarına göre en büyük 5 konturu sırala
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Orijinal görüntü üzerinde konturları çizmek için kopya oluştur
        self.img_contours = self.img.copy()
        # En büyük konturu yeşil renkle çiz
        cv2.drawContours(self.img_contours, contours[0], -1, (0, 255, 0), 3)
        
        return contours
        
    def array_sorting(self, pts):
        # Köşe noktalarını sıralamak için boş matris oluştur (4 nokta, her biri x,y)
        rect = np.zeros((4, 2), dtype="float32")

        # Her noktanın x+y toplamını hesapla
        s = pts.sum(axis=1)
        # En küçük toplamlı nokta sol üst, en büyük toplamlı nokta sağ alt olarak ayarla
        rect[0] = pts[np.argmin(s)] 
        rect[2] = pts[np.argmax(s)] 

        # Kalan iki noktayı seç (sol üst ve sağ alt hariç)
        arr = np.delete(pts, [np.argmin(s), np.argmax(s)], axis=0)
        column = arr[:, 0]
        # X değerine göre en büyük olan sağ üst, en küçük olan sol alt olur
        rect[1] = arr[np.argmax(column)]
        rect[3] = arr[np.argmin(column)]  
        return rect
    
    def perspective(self, rect):
        # Hedef perspektif dönüşüm sonrası genişlik ve yükseklik
        width, height = 400, 500
        # Hedef koordinatlar (düz dikdörtgen)
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # Perspektif dönüşüm matrisini hesapla
        matrix = cv2.getPerspectiveTransform(rect, pts2)
        # Görüntüyü perspektif dönüşümü ile düzelt
        self.warped = cv2.warpPerspective(self.img, matrix, (width, height))
    
    def corner_detection(self, contours):
        # En büyük konturun çevresini hesapla
        peri = cv2.arcLength(contours[0], True)  
        # Yaklaşım hassasiyetini belirle (kontur köşelerini daha doğru bulmak için)
        epsilon = 0.02 * peri                 
        # Konturu köşelere indirger
        approx = cv2.approxPolyDP(contours[0], epsilon, True)  
        
        # Bulunan köşe noktalarını listeye ekle ve görüntüye kırmızı daire çiz
        for point in approx:
            x, y = point[0] 
            self.tump.append((x, y))
            
            cv2.circle(self.img_contours, (x, y), radius=5, color=(0, 0, 255), thickness=-1) 
        
    def main(self):
        # Kenarları bul
        contours = self.edge_detection()
        # Köşe noktalarını tespit et
        self.corner_detection(contours)
        
        # Eğer 4 köşe bulunduysa perspektif dönüşümü uygula
        if 4 == len(self.tump):
            pts = np.array(self.tump, dtype="float32")
            rect = self.array_sorting(pts)
            self.perspective(rect)
            # Sonucu göster
            cv2.imshow('Perspektif', self.warped)
            # Sonucu dosyaya kaydet
            cv2.imwrite("output.jpg", self.warped)
        else:
            print("perspektif için 4 köşeye ihtiyaç vardır")
        
        # Kenarların ve köşelerin çizildiği orijinal görüntüyü göster
        cv2.imshow('Goruntu', self.img_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 'input.jpg' dosyasını oku
    img = cv2.imread('input.jpg')
    # Nesne oluştur ve işlemi başlat
    process = PerspectiveCorrection(img)
    process.main()

```
