import tensorflow as tf
# import tensorflow_text as text

# Load the model
model = tf.saved_model.load('translator')

# Use the model
# For example, to translate a Portuguese sentence to English
pt_sentence = "Ola"
en_sentence = model(pt_sentence)

print(f"Portuguese: {pt_sentence}")
print(f"English: {en_sentence}")
