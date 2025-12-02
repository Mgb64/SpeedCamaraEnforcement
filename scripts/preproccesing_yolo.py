import os
from multiprocessing import Pool, cpu_count
from PIL import Image

base_dir = './compartido/SPEED_CAMARA_ENFORCEMENT/DATASETS/CCPD2019'
subdirs = ["ccpd_base", "ccpd_db", "ccpd_blur", "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_challenge", "ccpd_weather"]

labels_root = os.path.join(base_dir, "labels")

def procesar_imagen(args):
    folder, filename = args
    img_path = os.path.join(base_dir, folder, filename)

    if not filename.endswith(".jpg"):
        return

    # Crear carpeta correspondiente dentro de labels
    label_folder = os.path.join(labels_root, folder)
    os.makedirs(label_folder, exist_ok=True)

    txt_path = os.path.join(label_folder, filename.replace(".jpg", ".txt"))

    try:
        parts = filename.split("-")
        box = parts[2]

        p1, p2 = box.split("_")
        xmin, ymin = map(int, p1.split("&"))
        xmax, ymax = map(int, p2.split("&"))

        # Leer imagen
        with Image.open(img_path) as img:
            w, h = img.size

        # Formato YOLO
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        with open(txt_path, "w") as f:
            f.write(f"0 {x_center} {y_center} {width} {height}")

    except Exception as e:
        print(f"Error procesando {img_path}: {e}")

def main():
    all_files = []
    for folder in subdirs:
        files = os.listdir(os.path.join(base_dir, folder))
        all_files.extend([(folder, f) for f in files])

    nproc = min(32, cpu_count())
    print(f"Procesando {len(all_files)} imágenes en {len(subdirs)} carpetas con {nproc} procesos…")

    with Pool(nproc) as p:
        p.map(procesar_imagen, all_files)

if __name__ == "__main__":
    main()
