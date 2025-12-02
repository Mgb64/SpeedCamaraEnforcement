import os
import shutil

# Base de CCPD2019
base_dir = "./compartido/SPEED_CAMARA_ENFORCEMENT/DATASETS/CCPD2019"

# Carpetas originales de imágenes y labels
img_base_dir = os.path.join(base_dir, "ccpd_base")
label_dir = os.path.join(base_dir, "labels")
split_dir = os.path.join(base_dir, "splits")

# Subcarpetas de test
test_subfolders = ["ccpd_db", "ccpd_blur", "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_challenge"]

# Carpetas destino para YOLO
yolo_base = os.path.join(base_dir, "yolo")
for sub in ["train", "val", "test"]:
    os.makedirs(os.path.join(yolo_base, "images", sub), exist_ok=True)
    os.makedirs(os.path.join(yolo_base, "labels", sub), exist_ok=True)

# Archivos de split
splits = {
    "train": os.path.join(split_dir, "train.txt"),
    "val": os.path.join(split_dir, "val.txt"),
    "test": os.path.join(split_dir, "test.txt"),
}

for split, file_path in splits.items():
    with open(file_path, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    print(f"Procesando {split}, total: {len(filenames)}")
    missing_labels = []

    for name in filenames:
        name_only = os.path.basename(name)
        
        # Determinar ruta de la imagen
        if split == "test":
            img_file = None
            for folder in test_subfolders:
                candidate = os.path.join(base_dir, folder, name_only)
                if os.path.exists(candidate):
                    img_file = candidate
                    break
            if img_file is None:
                print(f"Imagen de test no encontrada: {name_only}")
                continue
        else:
            img_file = os.path.join(img_base_dir, name_only)

        # Determinar ruta del label
        if split in ["train", "val"]:
            txt_file = os.path.join(label_dir, "ccpd_base", name_only.replace(".jpg", ".txt"))
            if not os.path.exists(txt_file):
                missing_labels.append(name_only.replace(".jpg", ".txt"))
        else:  # test
            txt_file = None
            for folder in test_subfolders:
                candidate = os.path.join(label_dir, folder, name_only.replace(".jpg", ".txt"))
                if os.path.exists(candidate):
                    txt_file = candidate
                    break
            if txt_file is None:
                missing_labels.append(name_only.replace(".jpg", ".txt"))

        # Copiar imagen
        if os.path.exists(img_file):
            shutil.copy(img_file, os.path.join(yolo_base, "images", split))
        else:
            print(f"Imagen no encontrada: {img_file}")

        # Copiar label
        if txt_file and os.path.exists(txt_file):
            shutil.copy(txt_file, os.path.join(yolo_base, "labels", split))

    if missing_labels:
        print(f"\n¡Atención! {len(missing_labels)} labels no se encontraron en {split}:")
        for l in missing_labels[:10]:
            print("Label no encontrado:", l)
        if len(missing_labels) > 10:
            print(f"... y {len(missing_labels)-10} más.\n")

    print(f"{split}: OK\n")

print("✅ Todo listo. Dataset YOLO creado en:", yolo_base)
