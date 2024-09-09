import os
import shutil

import torch
import yaml
import requests


def load_model(model_path, device):
    # URL publique pour télécharger les poids

    local_model_weights_path = './artefacts/simclr_pretrained.pth'

    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(local_model_weights_path), exist_ok=True)

    # Télécharger le fichier des poids
    response = requests.get(model_path)

    # Vérifier si le téléchargement a réussi
    if response.status_code == 200:
        with open(local_model_weights_path, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    # Charger les poids dans le modèle si le téléchargement a réussi
    if os.path.exists(local_model_weights_path):
        try:
            checkpoint = torch.load(local_model_weights_path, map_location=device)  # Remplacez 'cpu' par 'device' si nécessaire
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print("Model weights file does not exist.")
        
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    
    return state_dict


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(src=filename, dst='model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)
