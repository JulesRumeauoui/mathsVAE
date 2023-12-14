import librosa
import numpy as np
import os
import soundfile as sf
import resampy
import sounddevice as sd  # Assurez-vous d'installer la bibliothèque sounddevice si elle n'est pas déjà installée
import keyboard
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random

longueur = 80000
diminution = 1
samplerate = 0
dataEtLabel = []
np.set_printoptions(threshold=np.inf)  # Cela permet d'afficher l'intégralité du tableau sans aucune limitation



def sampling(args):
    z_mean, z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon
 
def play(data):
    global samplerate
    print(samplerate)
    sf.write('result.wav', data, samplerate, subtype='PCM_16')

    sd.play(data,samplerate)
    while sd.get_stream().active and not keyboard.is_pressed('space'):
        continue
    sd.stop()
    sf.read

def encode(file_name) :
    global samplerate
    # Charger le fichier audio
    current_directory = os.path.abspath(os.getcwd())
    print(current_directory)
    relative_path = os.path.join(current_directory, file_name)
    print(relative_path)
    data, samplerate = sf.read(relative_path, frames=100000000)
    samplerate = int(samplerate / diminution)

    # Rééchantillonner les données audio au taux d'échantillonnage cible
    print(data.shape)
    data = data[::diminution]  # Pour enlever un élément sur deux
    print(data.shape)


    return data

    # Vous pouvez ensuite utiliser la méthode np.save pour enregistrer le tableau
    # np.save('fichier_audio_spectrogram.npy', spectrogram_normalized)

def splitData(data) :
    
    split_data = [data[i:i + longueur, :] for i in range(0, data.shape[0], longueur)]


    # Vérifier si la taille du dernier morceau est différente de 1000000 et le retirer si c'est le cas
    if len(split_data[-1]) != longueur:
        split_data = split_data[:-1]
    # Vérifier les dimensions des morceaux découpés
    # for i, chunk in enumerate(split_data):
    #     print(f"Chunk {i + 1} : {chunk.shape}")
    play(split_data[5])


    return split_data



def getData(listeSon) :
    split_data = []
    labels = []
    for son in listeSon :
        file_name = son
        data = encode(file_name)
        data = splitData(data)
        for e in data:
            split_data.append(e)
            labels.append(son)  # Ajoutez l'étiquette ici pour chaque élément de data

    return split_data , labels



listeSon = ["feather.mp3","duhast.mp3"]
data, labels = getData(listeSon)

# random.shuffle(data)
data_size = len(data)
print(data_size)
# data = np.array(data)
# labels = np.array(labels)
# print(data.shape)
# print(labels.shape)
# print(labels)
pourcentage_data_train =  0.8
pourcentage_data_test =  0.2
pourcentage_data_evaluate =  0.0


batch_size = 1
epochs = 1  


taille_data_train = int(data_size * pourcentage_data_train)
taille_data_test = int(data_size * pourcentage_data_test)
taille_data_evaluate = int(data_size * pourcentage_data_evaluate)

print(taille_data_train)
print(taille_data_test)
print(taille_data_evaluate)
data_train = data[:taille_data_train] 
data_test = data[taille_data_train:taille_data_train+taille_data_test] 
data_evaluate = data[taille_data_evaluate:] 


target_train = data[:taille_data_train] 
target_test = data[taille_data_train:taille_data_train+taille_data_test] 
target_evaluate = data[taille_data_evaluate:] 

data_train = np.array(data_train)
data_test = np.array(data_test)
data_evaluate = np.array(data_evaluate)
target_train = np.array(target_train)
target_test = np.array(target_test)
target_evaluate = np.array(target_evaluate)

print("-----------")
print(data_train.shape)


latent_dim = 3

# Encodeur
inputs = keras.Input(shape=(longueur, 2))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation="tanh")(x)
x = layers.Dense(128, activation="tanh")(x)
x = layers.Dense(64, activation="tanh")(x)

z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Décodeur
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(64, activation="tanh")(latent_inputs)
x = layers.Dense(128, activation="tanh")(x)
x = layers.Dense(256, activation="tanh")(x)
x = layers.Dense(longueur * 2, activation="tanh")(x)
x = layers.Reshape((longueur, 2))(x)

decoder = keras.Model(latent_inputs, x, name="decoder")

# VAE
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name="vae")

# Loss personnalisée pour le VAE
xent_loss = longueur * 2 * keras.losses.mean_squared_error(inputs, outputs)
kl_loss = -0.5 * keras.backend.sum(1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var), axis=-1)
vae_loss = keras.backend.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer="adam")

history = vae.fit(data_train, target_train,
                 batch_size = batch_size,
                 epochs = epochs,
                 verbose = 1,
                 validation_data = (data_test, target_test),
                #  callbacks = callback_bestmodel
                 )


loss_curve = history.history["loss"]
val_loss_curve = history.history["val_loss"]
# acc_curve = history.history["accuracy"]
# val_acc_curve = history.history["val_accuracy"]

# plt.plot(loss_curve, label = 'loss')
# plt.plot(val_loss_curve, label = 'val_loss')
# plt.legend()
# plt.title("Loss")
# plt.show()

# plt.plot(acc_curve, label = 'acc')
# plt.plot(val_acc_curve, label = 'val_acc')
# plt.legend()
# plt.title("Acc")
# plt.show()

predictions = []
for sample in data:
    prediction = encoder.predict(np.expand_dims(sample, axis=0))
    predictions.append(prediction)
predictions = np.array(predictions)
print(predictions.shape)

flattened_predictions = np.squeeze(predictions)

# Diviser les coordonnées x et y pour créer des listes séparées
x_coords = flattened_predictions[:, 0]
y_coords = flattened_predictions[:, 1]
z_coords = flattened_predictions[:, 2]

# Créer un dictionnaire pour mapper les labels à des couleurs
label_color_map = {}
unique_labels = list(set(labels))
num_labels = len(unique_labels)
color_map = plt.get_cmap('viridis')  # Vous pouvez choisir une autre colormap si vous le souhaitez
for i, label in enumerate(unique_labels):
    label_color_map[label] = color_map(i / num_labels)

print(labels)
# Tracer les points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(x_coords)):
    ax.scatter(x_coords[i], y_coords[i], z_coords[i], c=label_color_map[labels[i]], label=labels[i])
# ax.scatter(x_coords, y_coords, z_coords, label = labels)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
handles = [plt.Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor=label_color_map[label]) for label in unique_labels]
plt.legend(handles, unique_labels)
plt.show(block=False)
plt.pause(0.001)

while True:
   

    # Demander les coordonnées à l'utilisateur
    x = float(input("Entrez la valeur de x : "))
    
    y = float(input("Entrez la valeur de y : "))
    z = float(input("Entrez la valeur de z : "))
    

    # Appeler le modèle pour décoder les coordonnées x et y
    decoded_data = decoder.predict(np.expand_dims((x, y, z), axis=0))  # Remplacez model_decode par votre fonction de décodage réelle
    decoded_data = decoded_data[0]
    decoded_data = np.array(decoded_data)
    print(decoded_data.shape)
    play(decoded_data)
    # Ajouter les données au tableau



















 