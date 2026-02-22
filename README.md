# Image Processing API

## Project Overview

This project implements a REST API for uploading and processing images.

Features:
- Upload JPG and PNG images
- Extract metadata (width, height, format, size)
- Generate small (128x128) and medium (512x512) thumbnails
- Store results in memory
- Provide statistics endpoint

---

## Installation Steps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Running the API

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000


---

## API Endpoints

Add this section:

```markdown
## API Endpoints

- POST /api/images – Upload image
- GET /api/images – List images
- GET /api/images/{image_id} – Get image record
- GET /api/images/{image_id}/thumbnails/{size} – Get thumbnail
- GET /api/stats – API statistics

## Example Usage

```bash
curl -X POST "http://127.0.0.1:8000/api/images" \
  -F "file=@image.jpg"


---

## Processing Pipeline Explanation

Add:

```markdown
## Processing Pipeline

1. Validate file type
2. Save original image
3. Verify image integrity
4. Extract metadata
5. Generate thumbnails
6. Store result in memory
7. Return JSON response