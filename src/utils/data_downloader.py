import os
import zipfile
from settings import DATA_DIR
import subprocess


def identify_file_type(file_path):
    if file_path.endswith(".zip"):
        return "zip"
    elif file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
        return "tgz"
    else:
        raise ValueError("Tipo de archivo no soportado. Solo se permiten .zip y .tgz")


def download_and_extract(url, name):
    if os.path.exists(f"{DATA_DIR}/{name}"):
        print(f"El dataset {name} ya existe. Omitiendo descarga.")
        return

    file_type = identify_file_type(url)
    file_path = f"{DATA_DIR}/{name}.{file_type}"

    print(f"Descargando dataset {name}...")
    subprocess.run(["wget", url, "-O", file_path], check=True)
    print("Descarga completada.")

    if file_type == "tgz":
        print(f"Descomprimiendo dataset {name}...")
        subprocess.run(
            ["tar", "-xzf", file_path, "-C", DATA_DIR],
            check=True,
        )
        subprocess.run(["rm", file_path], check=True)
    elif file_type == "zip":
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
            print("Descarga y extracción completadas.")

    # Remove the downloaded file after extraction
    os.remove(file_path)
