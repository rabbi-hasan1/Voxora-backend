
import os
import hmac
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import modal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download


# ==========================================
# Configuration
# ==========================================

APP_NAME = "banglavoice-tts"

MODEL_REPO = "EMTIAZZ/chatterbox-bangla-tts"
BASE_REPO = "ResembleAI/chatterbox"

MODEL_DIR = Path("/models/pretrained_models")

MODEL_FILES = [
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "conds.pt",
]

BANGLA_FILES = [
    "tokenizer.json",
    "t3_bangla_888k.safetensors",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banglavoice")


# ==========================================
# Modal Configuration
# ==========================================

modal_app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name(
    "banglavoice-models",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install_from_requirements("requirements.txt")
)


# ==========================================
# Model Download
# ==========================================

def download_model_files():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Preparing Chatterbox base models...")

    # Download original Chatterbox components
    for filename in MODEL_FILES:

        destination = MODEL_DIR / filename

        if destination.exists():
            logger.info("Already exists: %s", filename)
            continue

        logger.info("Downloading: %s", filename)

        downloaded = hf_hub_download(
            repo_id=BASE_REPO,
            filename=filename,
        )

        shutil.copy2(
            downloaded,
            destination,
        )

    # Download Bangla tokenizer and checkpoint
    for filename in BANGLA_FILES:

        destination = MODEL_DIR / filename

        if destination.exists():
            logger.info("Already exists: %s", filename)
            continue

        logger.info("Downloading Bangla asset: %s", filename)

        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
        )

        shutil.copy2(
            downloaded,
            destination,
        )

    logger.info("All model files are ready.")


# ==========================================
# Request Schema
# ==========================================

class TTSRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=1000,
    )

    api_secret: str = Field(
        min_length=1,
    )


# ==========================================
# Modal Web Application
# ==========================================

@modal_app.function(
    image=image,
    gpu="T4",
    volumes={
        "/models": model_volume,
    },
    secrets=[
        modal.Secret.from_name("banglavoice-secret"),
    ],
    timeout=600,
    scaledown_window=60,
    max_containers=1,
)
@modal.asgi_app()
def fastapi_app():

    from app import tts_service

    # ======================================
    # FastAPI lifecycle
    # ======================================

    @asynccontextmanager
    async def lifespan(application: FastAPI):

        logger.info("Preparing BanglaVoice model...")

        download_model_files()

        # Persist downloaded files
        model_volume.commit()

        logger.info("Model files are ready.")

        yield

        logger.info("Shutting down BanglaVoice.")

    web_app = FastAPI(
        title="BanglaVoice TTS API",
        description="Bangla text-to-speech powered by Chatterbox",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ======================================
    # Health Check
    # ======================================

    @web_app.get("/health")
    def health_check():

        return {
            "success": True,
            "service": APP_NAME,
            "model": MODEL_REPO,
            **tts_service.health(),
        }

    # ======================================
    # Generate Speech
    # ======================================

    @web_app.post("/generate")
    def generate_speech(payload: TTSRequest):

        expected_secret = os.environ.get(
            "TTS_API_SECRET",
            "",
        )

        if not expected_secret:
            raise HTTPException(
                status_code=500,
                detail="TTS API secret is not configured",
            )

        if not hmac.compare_digest(
            payload.api_secret,
            expected_secret,
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid API secret",
            )

        text = payload.text.strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty",
            )

        try:

            logger.info(
                "Generating Bangla speech. Characters: %s",
                len(text),
            )

            audio = tts_service.generate(
                text=text,
            )

            logger.info("Speech generated successfully.")

            return Response(
                content=audio,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": (
                        'attachment; filename="speech.wav"'
                    ),
                    "Cache-Control": "no-store",
                },
            )

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(error),
            )

        except Exception:

            logger.exception("TTS generation failed.")

            raise HTTPException(
                status_code=500,
                detail="Speech generation failed",
            )

    return web_app
