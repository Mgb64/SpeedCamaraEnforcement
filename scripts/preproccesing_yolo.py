import os
from multiprocessing import Pool, cpu_count
from PIL import Image

base_dir = './compartido/SPEED_CAMARA_ENFORCEMENT/DATASETS/CCPD2019'
img_dir = f"{base_dir}/ccpd_base/"
txt_dir = f"{base_dir}/labels/"
os.makedirs(txt_dir, exist_ok=True)

def procesar_imagen(filename):
    if not filename.endswith(".jpg"):
        return

    try:
        parts = filename.split("-")
        box = parts[2]  # ejemplo: '154&383_386&473'
        
        p1, p2 = box.split("_")
        xmin, ymin = map(int, p1.split("&"))
        xmax, ymax = map(int, p2.split("&"))

        # leer imagen
        img = Image.open(os.path.join(img_dir, filename))
        w, h = img.size

        # YOLO format
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        txt_name = filename.replace(".jpg", ".txt")
        with open(os.path.join(txt_dir, txt_name), "w") as f:
            f.write(f"0 {x_center} {y_center} {width} {height}")

    except Exception as e:
        # si alguna imagen está dañada o algo falla, no detiene todo
        print(f"Error procesando {filename}: {e}")

def main():
    files = os.listdir(img_dir)

    print(f"Procesando {len(files)} imágenes con {cpu_count()} CPUs…")

    with Pool(cpu_count()) as p:
        p.map(procesar_imagen, files)

if __name__ == "__main__":
    main()
