from transformers import AutoTokenizer, TFAutoModelForImageClassification
import tensorflow as tf

# Model ve tokenizer'ı yükleme
model_name = "nickmuchi/yolos-small-plant-disease-detection"
model = TFAutoModelForImageClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sınıf etiketleri
classes = ['healthy', 'diseased', 'powdery']

# Görüntü yolu
img_path = 'BHG-spider-plant-c0e0fdd5ec6e4c1588998ce3167f6579.jpg'

# Görüntüyü yükleme ve işleme
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)

# Tokenization
inputs = tokenizer(img_array, return_tensors="tf", padding=True, truncation=True, max_length=512)

# Tahmin
outputs = model(**inputs)
predicted_class_index = tf.argmax(outputs.logits[0]).numpy()
predicted_class_name = classes[predicted_class_index]

print("Tahmin edilen sınıf:", predicted_class_name)
