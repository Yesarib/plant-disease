from keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Eğittiğiniz modelin dosya yolunu belirtin
model_path = 'my_plant_disease_model.h5'

# Modeli yükleyin
model = load_model(model_path)

# Yeni görüntünün dosya yolunu belirtin
image_path = 'BHG-spider-plant-c0e0fdd5ec6e4c1588998ce3167f6579.jpg'

# Görüntüyü modelin beklentilerine uygun şekilde yeniden boyutlandırın ve ön işleme yapın
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Görüntüyü 0-1 aralığına ölçeklendirin

# Tahmin yapın
predictions = model.predict(img_array)

# Sınıfları ve tahmin olasılıklarını eşleyin
class_names = ["healthy", "powdery", "rust"]
result = dict(zip(class_names, predictions[0]))

# Tahmin sonuçlarını yazdırın
print(result)
