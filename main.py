import os
import re
import uuid
import shutil
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Создаём нужные папки при старте если их нет
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

VIN_PATTERN = re.compile(r'[A-HJ-NPR-Z0-9]{17}')

def extract_vins_from_pdf(pdf_path: str) -> List[str]:
    vins = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            found = VIN_PATTERN.findall(text)
            for vin in found:
                if vin not in vins:
                    vins.append(vin)
    return vins

def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def recognize_vin_on_photo(image_path: str) -> str | None:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(
            img,
            config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHJKLMNPRSTUVWXYZ0123456789'
        )
        text_clean = text.upper().replace(" ", "").replace("\n", "")
        matches = VIN_PATTERN.findall(text_clean)
        if matches:
            return matches[0]

        processed = preprocess_image(image_path)
        pil_img = Image.fromarray(processed)
        text2 = pytesseract.image_to_string(
            pil_img,
            config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHJKLMNPRSTUVWXYZ0123456789'
        )
        text2_clean = text2.upper().replace(" ", "").replace("\n", "")
        matches2 = VIN_PATTERN.findall(text2_clean)
        if matches2:
            return matches2[0]

    except Exception:
        pass
    return None

def cleanup_old_files(session_id: str):
    session_upload = UPLOAD_DIR / session_id
    session_result = RESULT_DIR / session_id
    if session_upload.exists():
        shutil.rmtree(session_upload)
    if session_result.exists():
        shutil.rmtree(session_result)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    photos: List[UploadFile] = File(...)
):
    session_id = str(uuid.uuid4())
    session_upload = UPLOAD_DIR / session_id
    session_upload.mkdir(parents=True)
    session_result = RESULT_DIR / session_id
    session_result.mkdir(parents=True)

    pdf_path = session_upload / "table.pdf"
    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())

    photos_dir = session_upload / "photos"
    photos_dir.mkdir()
    photo_names = []
    for photo in photos:
        dest = photos_dir / photo.filename
        with open(dest, "wb") as f:
            f.write(await photo.read())
        photo_names.append(photo.filename)

    photo_names.sort()

    pdf_vins = extract_vins_from_pdf(str(pdf_path))

    photo_vin_map = {}
    for name in photo_names:
        path = str(photos_dir / name)
        vin = recognize_vin_on_photo(path)
        photo_vin_map[name] = vin

    matched_vins = set()
    vin_to_photos = {}

    for i, name in enumerate(photo_names):
        vin = photo_vin_map.get(name)
        if vin and vin in pdf_vins:
            group = [name]
            if i + 1 < len(photo_names):
                group.append(photo_names[i + 1])
            if i + 2 < len(photo_names):
                group.append(photo_names[i + 2])
            vin_to_photos[vin] = group
            matched_vins.add(vin)

    output_dir = session_result / "Проба"
    output_dir.mkdir()

    for vin in pdf_vins:
        vin_dir = output_dir / vin
        vin_dir.mkdir()
        if vin in vin_to_photos:
            for photo_name in vin_to_photos[vin]:
                src = photos_dir / photo_name
                dst = vin_dir / photo_name
                shutil.copy2(src, dst)

    missing_vins = [v for v in pdf_vins if v not in matched_vins]

    zip_path = session_result / "Проба.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.rglob('*'):
            zf.write(file_path, file_path.relative_to(session_result))

    shutil.rmtree(session_upload)

    return JSONResponse({
        "session_id": session_id,
        "total_vins": len(pdf_vins),
        "matched": len(matched_vins),
        "missing_count": len(missing_vins),
        "missing_vins": missing_vins,
        "unrecognized_photos": [n for n, v in photo_vin_map.items() if v is None],
        "download_url": f"/download/{session_id}"
    })

@app.get("/download/{session_id}")
async def download(session_id: str, background_tasks: BackgroundTasks):
    zip_path = RESULT_DIR / session_id / "Проба.zip"
    if not zip_path.exists():
        return JSONResponse({"error": "Файл не найден"}, status_code=404)
    background_tasks.add_task(cleanup_old_files, session_id)
    return FileResponse(
        path=str(zip_path),
        filename="Проба.zip",
        media_type="application/zip"
    )
