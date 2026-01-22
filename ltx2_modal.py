import os
import string
import tempfile
import time
from pathlib import Path

import modal
from modal.exception import TimeoutError as ModalTimeoutError

MODEL_ID = "Lightricks/LTX-2"
DETAILER_REPO_ID = "Lightricks/LTX-2-19b-IC-LoRA-Detailer"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
APP_NAME = "ltx2-text-to-video"

CHECKPOINT_FILENAME = "ltx-2-19b-dev.safetensors"
DISTILLED_LORA_FILENAME = "ltx-2-19b-distilled-lora-384.safetensors"
DETAILER_LORA_FILENAME = "ltx-2-19b-ic-lora-detailer.safetensors"
SPATIAL_UPSAMPLER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Install LTX-2 core/pipeline packages from the local subtree.
    .add_local_dir("LTX-2", "/root/LTX-2", copy=True)
    .uv_pip_install(
        "accelerate==1.6.0",
        "av==12.0.0",
        "einops==0.8.0",
        "huggingface-hub==0.36.0",
        "numpy==2.0.2",
        "pillow==11.1.0",
        "safetensors==0.5.2",
        "scipy==1.15.1",
        "sentencepiece==0.2.0",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "tqdm==4.67.1",
        "transformers==4.51.3",
        "fastapi[standard]",
        "file:///root/LTX-2/packages/ltx-core",
        "file:///root/LTX-2/packages/ltx-pipelines",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App(APP_NAME, secrets=[modal.Secret.from_name("HF_TOKEN")])

VOLUME_NAME = "ltx2-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")
MODEL_VOLUME_NAME = "ltx2-model-without-transformers"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

MODEL_PATH = Path("/models")
image = image.env({"HF_HOME": str(MODEL_PATH)})


MINUTES = 60  # seconds

FRAME_RATE = 24.0

def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    return None

def slugify(prompt):
    for char in string.punctuation:
        prompt = prompt.replace(char, "")
    prompt = prompt.replace(" ", "_")
    prompt = prompt[:230]  # some OSes limit filenames to <256 chars
    mp4_name = str(int(time.time())) + "_" + prompt + ".mp4"
    return mp4_name

def get_num_frames(seconds: int) -> int:
    raw_frames = seconds * FRAME_RATE
    return max(1, int(raw_frames + 1))

def build_generation_kwargs(
    *,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seconds: int,
    width: int,
    height: int,
    seed: int,
    use_detailer_lora: bool,
    image_bytes: bytes | None,
    image_filename: str,
    image_strength: float,
) -> dict:
    num_frames = get_num_frames(seconds)
    return dict(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        width=width,
        height=height,
        frame_rate=FRAME_RATE,
        seed=seed,
        use_detailer_lora=use_detailer_lora,
        image_bytes=image_bytes,
        image_filename=image_filename,
        image_strength=image_strength,
    )

def fetch_image_from_url(image_url: str) -> tuple[bytes, str]:
    from urllib.parse import urlparse
    from urllib.request import urlopen

    try:
        with urlopen(image_url, timeout=20) as response:
            image_bytes = response.read()
    except Exception as exc:
        raise ValueError("Failed to fetch image_url") from exc
    parsed = urlparse(image_url)
    filename = Path(parsed.path).name or "image.png"
    return image_bytes, filename

def read_local_image(image_path: str) -> tuple[bytes, str]:
    path = Path(image_path)
    return path.read_bytes(), path.name

def resolve_image_input(image_path: str | None, image_url: str | None) -> tuple[bytes | None, str]:
    if image_path and image_url:
        raise ValueError("Provide image_path or image_url, not both")
    if image_path:
        return read_local_image(image_path)
    if image_url:
        return fetch_image_from_url(image_url)
    return None, "image.png"

def try_get_call_result(call: modal.FunctionCall) -> str | None:
    try:
        return call.get(timeout=0)
    except (ModalTimeoutError, TimeoutError):
        return None


@app.cls(
    image=image,  # use our container Image
    volumes={OUTPUTS_PATH: outputs, MODEL_PATH: model},  # attach our Volumes
    gpu="H100",  # use a big, fast GPU
    timeout=10 * MINUTES,  # run inference for up to 10 minutes
    scaledown_window=1 * MINUTES,  # stay idle for 1 minute before scaling down
)


class LTX2:
    @modal.enter()
    def load_model(self):
        from huggingface_hub import hf_hub_download, snapshot_download

        model_dir = MODEL_PATH / "ltx2"
        model_dir.mkdir(parents=True, exist_ok=True)
        token = get_hf_token()
        self.checkpoint_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=CHECKPOINT_FILENAME,
            cache_dir=str(MODEL_PATH),
            local_dir=str(model_dir),
            token=token,
        )
        self.distilled_lora_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=DISTILLED_LORA_FILENAME,
            cache_dir=str(MODEL_PATH),
            local_dir=str(model_dir),
            token=token,
        )
        self.spatial_upsampler_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=SPATIAL_UPSAMPLER_FILENAME,
            cache_dir=str(MODEL_PATH),
            local_dir=str(model_dir),
            token=token,
        )
        self.detailer_lora_path = hf_hub_download(
            repo_id=DETAILER_REPO_ID,
            filename=DETAILER_LORA_FILENAME,
            cache_dir=str(MODEL_PATH),
            local_dir=str(model_dir),
            token=token,
        )

        gemma_root = MODEL_PATH / "gemma"
        snapshot_download(
            repo_id=GEMMA_REPO_ID,
            cache_dir=str(MODEL_PATH),
            local_dir=str(gemma_root),
            token=token,
            allow_patterns=[
                "model*.safetensors",
                "model.safetensors.index.json",
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
            ],
        )
        self.gemma_root = str(gemma_root)
        self.pipeline = None
        self._pipeline_scales = None

    @modal.method()
    def generate(
        self,
        prompt,
        num_inference_steps=40,
        num_frames=121,
        width=1536,
        height=1024,
        frame_rate=FRAME_RATE,
        guidance_scale=4.0,
        seed=42,
        use_detailer_lora=False,
        image_bytes: bytes | None = None,
        image_filename: str = "image.png",
        image_strength: float = 1.0,
    ):
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT
        from ltx_pipelines.utils.media_io import encode_video

        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

        loras = []
        distilled_loras = [LoraPathStrengthAndSDOps(self.distilled_lora_path, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)]
        if use_detailer_lora:
            detailer = LoraPathStrengthAndSDOps(
                self.detailer_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP
            )
            loras.append(detailer)
            distilled_loras.append(detailer)

        pipeline_key = (
            "detailer-on" if use_detailer_lora else "detailer-off",
            "distilled-0.8",
        )
        if self.pipeline is None or self._pipeline_scales != pipeline_key:
            self.pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=distilled_loras,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=self.gemma_root,
                loras=loras,
                device="cuda",
            )
            self._pipeline_scales = pipeline_key

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        images = []
        if image_bytes:
            suffix = Path(image_filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(image_bytes)
                images.append((tmp_file.name, 0, image_strength))

        print("🧠 LTX2: starting two-stage pipeline")
        pipeline_start = time.time()
        video, audio = self.pipeline(
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=guidance_scale,
            images=images,
            tiling_config=tiling_config,
        )
        pipeline_elapsed = time.time() - pipeline_start
        print(f"🧠 LTX2: pipeline complete in {pipeline_elapsed:.2f}s")

        import torch

        mp4_name = slugify(prompt)
        with torch.inference_mode():
            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(Path(OUTPUTS_PATH) / mp4_name),
                video_chunks_number=video_chunks_number,
            )

        outputs.commit()
        return mp4_name

web_app = None

def build_web_app():
    from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
    from fastapi.responses import StreamingResponse
    from modal.exception import NotFoundError

    app = FastAPI()

    def get_call_or_404(job_id: str) -> modal.FunctionCall:
        try:
            return modal.FunctionCall.from_id(job_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown job_id") from exc

    @app.post("/generate")
    async def generate_video(
        request: Request,
        prompt: str | None = Form(None),
        num_inference_steps: int = Form(40),
        guidance_scale: float = Form(4.0),
        seconds: int = Form(5),
        width: int = Form(1536),
        height: int = Form(1024),
        seed: int = Form(42),
        use_detailer_lora: bool = Form(False),
        image_strength: float = Form(1.0),
        image_url: str | None = Form(None),
        image_path: UploadFile | None = File(None),
    ):
        ltx2 = LTX2()
        if request.headers.get("content-type", "").startswith("application/json"):
            raise HTTPException(status_code=400, detail="Use multipart form data")
        if prompt is None:
            raise HTTPException(status_code=400, detail="Missing prompt")
        image_bytes = None
        image_filename = "image.png"
        if image_path is not None and image_url:
            raise HTTPException(status_code=400, detail="Provide image_path or image_url, not both")
        if image_path is not None:
            image_bytes = await image_path.read()
            image_filename = image_path.filename or image_filename
        elif image_url:
            try:
                image_bytes, image_filename = fetch_image_from_url(image_url)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        generation_kwargs = build_generation_kwargs(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seconds=seconds,
            width=width,
            height=height,
            seed=seed,
            use_detailer_lora=use_detailer_lora,
            image_bytes=image_bytes,
            image_filename=image_filename,
            image_strength=image_strength,
        )
        call = ltx2.generate.spawn(**generation_kwargs)
        job_id = getattr(call, "object_id", None) or str(call)
        base_url = str(request.base_url).rstrip("/")
        return {
            "job_id": job_id,
            "status_url": f"{base_url}/generate/{job_id}",
            "result_url": f"{base_url}/generate/{job_id}",
        }

    @app.head("/generate/{job_id}")
    def status_video(job_id: str):
        call = get_call_or_404(job_id)
        mp4_name = try_get_call_result(call)
        if mp4_name is None:
            return Response(status_code=202)
        return Response(status_code=200)

    @app.get("/generate/{job_id}")
    def get_video(job_id: str):
        call = get_call_or_404(job_id)
        mp4_name = try_get_call_result(call)
        if mp4_name is None:
            return Response(status_code=202)
        return StreamingResponse(outputs.read_file(mp4_name), media_type="video/mp4")

    return app

def get_web_app():
    global web_app
    if web_app is None:
        web_app = build_web_app()
    return web_app

@app.function(image=image, volumes={OUTPUTS_PATH: outputs})
@modal.asgi_app(requires_proxy_auth=True)
def api():
    return get_web_app()

@app.local_entrypoint()
def main(
    prompt="An animated polar bear walks into an igloo and says 'I'm home! Who is ready to party?'",
    num_inference_steps: int = 40,  
    guidance_scale: float = 4.0,
    seconds: int = 5,
    width: int = 1536,
    height: int = 1024,
    seed: int = 42,
    use_detailer_lora: bool = False,
    image_path: str = "",
    image_url: str = "",
    image_strength: float = 1.0,
    ):

    ltx2 = LTX2()

    def run():
        print(f"🎥 Generating a video from the prompt '{prompt}'")
        start = time.time()
        num_frames = get_num_frames(seconds)
        print(f"🎥 Using {num_frames} frames for {seconds}s at {FRAME_RATE:.0f} fps")
        try:
            image_bytes, image_filename = resolve_image_input(
                image_path or None,
                image_url or None,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        generation_kwargs = build_generation_kwargs(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seconds=seconds,
            width=width,
            height=height,
            seed=seed,
            use_detailer_lora=use_detailer_lora,
            image_bytes=image_bytes,
            image_filename=image_filename,
            image_strength=image_strength,
        )
        call = ltx2.generate.spawn(**generation_kwargs)
        job_id = getattr(call, "object_id", None) or str(call)
        print(f"🎥 Job id: {job_id}")

        mp4_name = call.get()

        duration = time.time() - start
        print(f"🎥 Client received video in {int(duration)}s")
        print(f"🎥 LTX2 video saved to Modal Volume at {mp4_name}")

        local_dir = Path("outputs")
        local_dir.mkdir(exist_ok=True, parents=True)
        local_path = local_dir / mp4_name
        local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
        print(f"🎥 LTX2 video saved locally at {local_path}")

    run()
