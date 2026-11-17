from pathlib import Path
import cv2
from ultralytics.data.augment import LetterBox

src = Path(r"E:\Desktop\changzhou\dataset\png_yolo_8_2\images\val")
dst = Path(r"E:\Desktop\changzhou\black\yolov8_bst_sample_bsnnrt\yolov8_bst_sample\train\compiler\calibrate_data")
dst.mkdir(parents=True, exist_ok=True)

files = sorted(
    p for p in src.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)

letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleup=False)

saved = 0
for path in files:
    if saved >= 50:
        break

    image = cv2.imread(str(path))
    if image is None:
        continue

    image = letterbox(image=image)
    cv2.imwrite(str(dst / f"{saved:04d}.jpg"), image)
    saved += 1

print(f"saved {saved} calibration images")