import pandas as pd
import csv
import os


def create_metrics_csv(path: str):
    # Obtenez le répertoire du fichier
    directory = os.path.dirname(path)

    # Créez les répertoires manquants
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "val_loss", "accuracy", "val_accuracy"])


def append_metrics_csv(
    path: str, loss: float, accuracy: float, val_loss: float, val_accuracy: float
):
    with open(path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([loss, val_loss, accuracy, val_accuracy])


def get_number_of_epochs(log_path:str):
        metrics = pd.read_csv(log_path)
        return metrics.shape[0]