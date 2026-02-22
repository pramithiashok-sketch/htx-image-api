# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from PIL import Image, UnidentifiedImageError
import time
import logging


import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

IMAGES: dict[str, dict] = {}

ALLOWED_TYPES = {"image/jpeg": "jpg", "image/png": "png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB

# ---------- LOGGING ----------
logger = logging.getLogger("image_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
processor: BlipProcessor | None = None
model: BlipForConditionalGeneration | None = None


@app.on_event("startup")
def _load_caption_model() -> None:
    """
    Load once when FastAPI starts (better with uvicorn --reload and subprocesses).
    """
    global processor, model
    logger.info("Loading caption model: %s", BLIP_MODEL_ID)
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    model.eval()
    logger.info("Caption model loaded.")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_format(pil_format: str | None, fallback_ext: str) -> str:
    """
    Pillow gives "JPEG" / "PNG". Spec examples often use "jpg" / "png".
    """
    fmt = (pil_format or fallback_ext).upper()
    if fmt == "JPEG":
        return "jpg"
    return fmt.lower()


def caption_image(image_path: Path) -> str:
    """
    Returns a short caption for an image using BLIP.
    Never raises (returns a string describing the failure).
    """
    global processor, model
    if processor is None or model is None:
        return "caption not ready"

    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=25,
                num_beams=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return processor.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        logger.exception("Captioning failed")
        return f"caption exception: {type(e).__name__}"


def _make_failed_record(image_id: str, filename: str, processed_at: str, error: str) -> dict:
    rec = {
        "status": "failed",
        "data": {
            "image_id": image_id,
            "original_name": filename,
            "processed_at": processed_at,
            "metadata": {},
            "thumbnails": {},
        },
        "error": error,
    }
    IMAGES[image_id] = rec
    return rec


@app.get("/")
def root():
    return {"message": "Server is running"}


@app.post("/api/images")
async def upload_image(request: Request, file: UploadFile = File(...)):
    t0 = time.perf_counter()
    processed_at = _now_utc_iso()

    # Validate 
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPG and PNG are supported")

    image_id = f"img{uuid4().hex[:8]}"
    ext = ALLOWED_TYPES[file.content_type]
    saved_path = UPLOAD_DIR / f"{image_id}.{ext}"

    logger.info(
        "Receiving upload: filename=%s content_type=%s image_id=%s",
        file.filename, file.content_type, image_id
    )

    
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES} bytes)")

    saved_path.write_bytes(data)

    # Validate a real image
    try:
        with Image.open(saved_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError):
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Gather metadata and make captions and Thumbnails
    try:
        with Image.open(saved_path) as img:
            width, height = img.size
            format_name = _normalize_format(img.format, ext)

        stat = saved_path.stat()
        size_bytes = stat.st_size
        file_mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        metadata = {
            "width": width,
            "height": height,
            "format": format_name,
            "size_bytes": size_bytes,
            "file_mtime_utc": file_mtime_utc,
        }

        
        small_path = UPLOAD_DIR / f"{image_id}_small.{ext}"
        medium_path = UPLOAD_DIR / f"{image_id}_medium.{ext}"

        
        with Image.open(saved_path) as img:
            if ext == "jpg" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((128, 128))
            img.save(small_path)

        
        with Image.open(saved_path) as img:
            if ext == "jpg" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((512, 512))
            img.save(medium_path)

        base = str(request.base_url).rstrip("/")
        thumbnails = {
            "small": f"{base}/api/images/{image_id}/thumbnails/small",
            "medium": f"{base}/api/images/{image_id}/thumbnails/medium",
        }

        
        caption = caption_image(saved_path)
        analysis = {"caption": caption}

        elapsed = time.perf_counter() - t0

        rec = {
            "status": "success",
            "data": {
                "image_id": image_id,
                "original_name": file.filename,
                "processed_at": processed_at,
                "metadata": metadata,
                "thumbnails": thumbnails,
                "analysis": analysis,
                "processing_time_seconds": round(elapsed, 4),
            },
            "error": None,
        }

        IMAGES[image_id] = rec
        logger.info("Processed image_id=%s in %.4fs", image_id, elapsed)
        return rec

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing failed for image_id=%s", image_id)

        
        for p in [
            saved_path,
            UPLOAD_DIR / f"{image_id}_small.{ext}",
            UPLOAD_DIR / f"{image_id}_medium.{ext}",
        ]:
            p.unlink(missing_ok=True)

        elapsed = time.perf_counter() - t0
        rec = _make_failed_record(image_id, file.filename, processed_at, f"{type(e).__name__}")
        rec["data"]["processing_time_seconds"] = round(elapsed, 4)
        return rec


@app.get("/api/images")
def list_images():
    return list(IMAGES.values())


@app.get("/api/images/{image_id}")
def get_image(image_id: str):
    if image_id not in IMAGES:
        raise HTTPException(status_code=404, detail="image not found")
    return IMAGES[image_id]


@app.get("/api/images/{image_id}/thumbnails/{size}")
def get_thumbnail(image_id: str, size: str):
    if image_id not in IMAGES:
        raise HTTPException(status_code=404, detail="image not found")

    if size not in {"small", "medium"}:
        raise HTTPException(status_code=400, detail="size must be small or medium")

    fmt = IMAGES[image_id].get("data", {}).get("metadata", {}).get("format", "jpg")
    ext = "png" if fmt == "png" else "jpg"

    thumb_path = UPLOAD_DIR / f"{image_id}_{size}.{ext}"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="thumbnail not found")

    return FileResponse(thumb_path)


@app.get("/api/stats")
def stats():
    total = len(IMAGES)
    failed = sum(1 for r in IMAGES.values() if r.get("status") == "failed")
    succeeded = sum(1 for r in IMAGES.values() if r.get("status") == "success")

    times = [
        r.get("data", {}).get("processing_time_seconds")
        for r in IMAGES.values()
        if r.get("status") == "success"
        and r.get("data", {}).get("processing_time_seconds") is not None
    ]
    avg = (sum(times) / len(times)) if times else 0.0

    success_rate = f"{(succeeded / total * 100):.2f}%" if total > 0 else "0.00%"

    return {
        "total": total,
        "failed": failed,
        "success_rate": success_rate,
        "average_processing_time_seconds": round(avg, 4),
    }

