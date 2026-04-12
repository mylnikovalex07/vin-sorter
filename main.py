import os
import re
import uuid
import shutil
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pdfplumber
import easyocr
from PIL import Image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# EasyOCR reader (загружается один раз)
reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

VIN_PATTERN = re.compile(r'[A-HJ-NPR-Z0-9]{17}')

def extract_vins_from_pdf(pdf_path: str) -> List[str]:
    """Извлекает все VIN-коды из PDF таблицы"""
    vins = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            found = VIN_PATTERN.findall(text)
            for vin in found:
                if vin not in vins:
                    vins.append(vin)
    return vins

def recognize_vin_on_photo(image_path: str) -> str | None:
    """Распознаёт VIN-код на фотографии через OCR"""
    try:
        ocr = get_reader()
        results = ocr.readtext(image_path)
        for (_, text, confidence) in results:
            text_clean = text.upper().replace(" ", "").replace("-", "")
            matches = VIN_PATTERN.findall(text_clean)
            if matches:
                return matches[0]
    except Exception:
        pass
    return None

def cleanup_old_files(session_id: str):
    """Удаляет временные файлы сессии"""
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

    # Сохраняем PDF
    pdf_path = session_upload / "table.pdf"
    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())

    # Сохраняем фото
    photos_dir = session_upload / "photos"
    photos_dir.mkdir()
    photo_names = []
    for photo in photos:
        dest = photos_dir / photo.filename
        with open(dest, "wb") as f:
            f.write(await photo.read())
        photo_names.append(photo.filename)

    # Сортируем фото по имени
    photo_names.sort()

    # 1. Извлекаем VIN-коды из PDF
    pdf_vins = extract_vins_from_pdf(str(pdf_path))

    # 2. Сканируем каждое фото на наличие VIN
    photo_vin_map = {}  # filename -> vin или None
    for name in photo_names:
        path = str(photos_dir / name)
        vin = recognize_vin_on_photo(path)
        photo_vin_map[name] = vin

    # 3. Раскладываем фото по папкам
    # Ищем фото с VIN и берём +2 следующих по имени
    matched_vins = set()
    vin_to_photos = {}  # vin -> [photo1, photo2, photo3]

    for i, name in enumerate(photo_names):
        vin = photo_vin_map.get(name)
        if vin and vin in pdf_vins:
            # Берём это фото + 2 следующих
            group = [name]
            if i + 1 < len(photo_names):
                group.append(photo_names[i + 1])
            if i + 2 < len(photo_names):
                group.append(photo_names[i + 2])
            vin_to_photos[vin] = group
            matched_vins.add(vin)

    # 4. Создаём папки и копируем файлы
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

    # 5. Определяем VIN без фото
    missing_vins = [v for v in pdf_vins if v not in matched_vins]

    # 6. Упаковываем в ZIP
    zip_path = session_result / "Проба.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.rglob('*'):
            zf.write(file_path, file_path.relative_to(session_result))

    # Удаляем временные загрузки
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
