import os
import tempfile
from dataclasses import dataclass

import modal
from fastapi import Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("LTX2_MODEL_ID", "Lightricks/LTX-2")
MODEL_CACHE_PATH = "/model-cache"
APP_NAME = "ltx2-text-to-video"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "accelerate>=0.30.0",
        "diffusers>=0.31.0",
        "safetensors>=0.4.0",
        "torch>=2.3.0",
        "transformers>=4.45.0",
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("ltx2-model-cache", create_if_missing=True)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to render into a video.")
    negative_prompt: str | None = Field(
        None, description="Optional negative prompt for classifier-free guidance."
    )
    height: int = Field(576, ge=256, le=1024)
    width: int = Field(1024, ge=256, le=1024)
    num_frames: int = Field(48, ge=8, le=160)
    fps: int = Field(24, ge=1, le=60)
    guidance_scale: float = Field(3.0, ge=1.0, le=15.0)
    num_inference_steps: int = Field(30, ge=1, le=100)
    seed: int = Field(0, ge=0)


@dataclass
class GenerationResult:
    video_bytes: bytes


@app.cls(
    gpu=modal.gpu.A10G(),
    image=image,
    timeout=60 * 20,
    container_idle_timeout=60 * 10,
    volumes={MODEL_CACHE_PATH: volume},
)
class LTX2Service:
    def __enter__(self):
        os.environ.setdefault("HF_HOME", MODEL_CACHE_PATH)
        from diffusers import LTX2Pipeline
        import torch

        self.torch = torch
        self.pipe = LTX2Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE_PATH,
        )
        self.pipe.to("cuda")

    @modal.method()
    def generate(self, payload: GenerateRequest) -> GenerationResult:
        generator = self.torch.Generator(device="cuda").manual_seed(payload.seed)
        result = self.pipe(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            height=payload.height,
            width=payload.width,
            num_frames=payload.num_frames,
            guidance_scale=payload.guidance_scale,
            num_inference_steps=payload.num_inference_steps,
            generator=generator,
        )

        from diffusers.utils import export_to_video

        frames = result.frames[0]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = export_to_video(frames, tmp_file.name, fps=payload.fps)

        with open(video_path, "rb") as handle:
            video_bytes = handle.read()

        os.remove(video_path)
        return GenerationResult(video_bytes=video_bytes)


def _resolve_token(authorization: str | None, x_api_token: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization.removeprefix("Bearer ").strip()
    if x_api_token:
        return x_api_token
    return None


@app.function(secrets=[modal.Secret.from_name("ltx2-api-token")])
@modal.web_endpoint(method="POST")
def generate_api(
    payload: GenerateRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
):
    expected = os.environ.get("LTX2_API_TOKEN")
    if not expected:
        raise HTTPException(status_code=500, detail="Server token is not configured.")

    provided = _resolve_token(authorization, x_api_token)
    if not provided:
        raise HTTPException(status_code=401, detail="Missing API token.")
    if provided != expected:
        raise HTTPException(status_code=403, detail="Invalid API token.")

    result = LTX2Service().generate.remote(payload)
    return Response(content=result.video_bytes, media_type="video/mp4")


@app.local_entrypoint()
def main(
    prompt: str,
    output_path: str = "ltx2-output.mp4",
    height: int = 576,
    width: int = 1024,
    num_frames: int = 48,
    fps: int = 24,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 30,
    seed: int = 0,
):
    payload = GenerateRequest(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    result = LTX2Service().generate.remote(payload)
    with open(output_path, "wb") as handle:
        handle.write(result.video_bytes)
    print(f"Saved video to {output_path}")
