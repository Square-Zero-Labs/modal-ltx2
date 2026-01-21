import os
import string
import time
from pathlib import Path

import modal

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
        width=768,
        height=512,
        frame_rate = 24.0,
        guidance_scale=4.0,
        seed=42,
    ):
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT
        from ltx_pipelines.utils.media_io import encode_video

        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

        loras = [LoraPathStrengthAndSDOps(self.detailer_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
        distilled_loras = [LoraPathStrengthAndSDOps(self.distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]

        pipeline_key = ("detailer-1.0", "distilled-1.0")
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

        print("🧠 LTX2: starting two-stage pipeline")
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
            images=[],
            tiling_config=tiling_config,
        )
        print("🧠 LTX2: pipeline complete")

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

@app.local_entrypoint()
def main(
    prompt="An animated polar bear walks into an igloo and says 'I'm home! Who is ready to party?'",
    num_inference_steps: int = 40,  
    guidance_scale: float = 4.0,
    seconds: int = 5,
    width: int = 768,
    height: int = 512,
    seed: int = 42,
    ):

    ltx2 = LTX2()

    def run():
        print(f"🎥 Generating a video from the prompt '{prompt}'")
        start = time.time()
        num_frames = seconds * 24
        mp4_name = ltx2.generate.remote(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed,
        )
        duration = time.time() - start
        print(f"🎥 Client received video in {int(duration)}s")
        print(f"🎥 LTX2 video saved to Modal Volume at {mp4_name}")

        local_dir = Path("outputs")
        local_dir.mkdir(exist_ok=True, parents=True)
        local_path = local_dir / mp4_name
        local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
        print(f"🎥 LTX2 video saved locally at {local_path}")

    run()
