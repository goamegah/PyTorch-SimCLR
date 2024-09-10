import os
import shutil

import torch
import yaml
import requests

import boto3
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET')
AWS_S3_BUCKET_OBJECT_NAME = os.getenv('AWS_S3_BUCKET_OBJECT_NAME')

def load_checkpoint(model_path, device, out='checkpoint.pth.tar'):
    # URL publique pour télécharger les poids

    ckpt_path = f'./artefacts/{out}'

    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Télécharger le fichier des poids
    response = requests.get(model_path)

    # Vérifier si le téléchargement a réussi
    if response.status_code == 200:
        with open(ckpt_path, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    # Charger les poids dans le modèle si le téléchargement a réussi
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)  # Remplacez 'cpu' par 'device' si nécessaire
            print("Model checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print("Model weights file does not exist.")
    
    return checkpoint


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(src=filename, dst='model_best.pth.tar')


def save_checkpoint_on_s3(state, is_best, filename='checkpoint.pth.tar'):
    # Save the checkpoint locally
    checkpoint_path = f'./artefacts/{filename}'
    torch.save(state, checkpoint_path)

    # Upload the checkpoint to S3
    s3_client = boto3.client('s3',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    try:
        s3_client.upload_file(checkpoint_path, AWS_S3_BUCKET_NAME, f'{AWS_S3_BUCKET_OBJECT_NAME}/{filename}')
        print(f"Checkpoint {filename} uploaded to S3 bucket {AWS_S3_BUCKET_NAME} successfully.")
        if is_best:
            s3_client.upload_file(checkpoint_path, AWS_S3_BUCKET_NAME, f'{AWS_S3_BUCKET_OBJECT_NAME}/model_best.pth.tar')
            print(f"Best model checkpoint uploaded to S3 bucket {AWS_S3_BUCKET_NAME} successfully.")
    except Exception as e:
        print(f"Failed to upload checkpoint to S3: {e}")


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)
