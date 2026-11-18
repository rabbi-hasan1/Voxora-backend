
import io
import os
import torch
import soundfile as sf

from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.t3.t3 import T3


MODEL_REPO = "EMTIAZZ/chatterbox-bangla-tts"

MODEL_DIR = Path(
    os.getenv("MODEL_DIR", "/models/pretrained_models")
)


class BanglaTTS:

    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tts = None

    def download_checkpoint(self):

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="t3_bangla_888k.safetensors",
            local_dir=str(MODEL_DIR)
        )

        return checkpoint

    def load_model(self):

        if self.tts is not None:
            return

        print("Loading Chatterbox base model...")

        self.tts = ChatterboxTTS.from_pretrained(
            device=self.device
        )

        print("Loading Bangla checkpoint...")

        checkpoint = self.download_checkpoint()

        # Update tokenizer vocabulary size
        self.tts.t3.hp.text_tokens_dict_size = 4240

        new_t3 = T3(hp=self.tts.t3.hp)

        state_dict = load_file(
            checkpoint,
            device="cpu"
        )

        # Handle checkpoint key prefixes
        if any(k.startswith("t3.") for k in state_dict):
            state_dict = {
                k.removeprefix("t3."): v
                for k, v in state_dict.items()
                if k.startswith("t3.")
            }

        new_t3.load_state_dict(
            state_dict,
            strict=True
        )

        self.tts.t3 = new_t3

        self.tts.t3.to(self.device).eval()
        self.tts.s3gen.to(self.device).eval()
        self.tts.ve.to(self.device).eval()

        print("Bangla TTS model loaded successfully!")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        audio_prompt_path: str | None = None
    ):

        if self.tts is None:
            self.load_model()

        text = text.strip()

        if not text:
            raise ValueError("Text cannot be empty")

        if len(text) > 1000:
            raise ValueError("Maximum 1000 characters")

        kwargs = {}

        if audio_prompt_path:
            if not os.path.isfile(audio_prompt_path):
                raise FileNotFoundError(
                    "Reference audio not found"
                )

            kwargs["audio_prompt_path"] = audio_prompt_path

        wav = self.tts.generate(
            text=text,
            **kwargs
        )

        audio_buffer = io.BytesIO()

        sf.write(
            audio_buffer,
            wav.squeeze().cpu().numpy(),
            self.tts.sr,
            format="WAV"
        )

        audio_buffer.seek(0)

        return audio_buffer.getvalue()

    def health(self):

        return {
            "status": "ready" if self.tts else "not_loaded",
            "device": self.device,
            "gpu_available": torch.cuda.is_available()
        }


# Singleton instance
tts_service = BanglaTTS()
