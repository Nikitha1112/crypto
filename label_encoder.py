from sklearn.preprocessing import LabelEncoder
import joblib

# List of algorithms your model predicts (replace with your actual class labels)
algorithms = [
    'AES', 'DES', 'Triple DES', 'RSA', 'Blowfish', 'ChaCha20',
    'RC4', 'Camellia', 'Serpent', 'ElGamal', 'ECC', 'GOST'
]

# Create the label encoder
label_encoder = LabelEncoder()

# Fit the encoder to the list of algorithms
label_encoder.fit(algorithms)

# Save the label encoder to a file
joblib.dump(label_encoder, 'label_encoder.joblib')

# Optionally, print the classes to verify
print(label_encoder.classes_)
