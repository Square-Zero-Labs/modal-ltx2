import gc
import string
import time
from pathlib import Path

import modal

MODEL_ID = "Lightricks/LTX-2"
APP_NAME = "ltx2-text-to-video"
STAGE_2_DISTILLED_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]

image = (
    modal.Image.debian_slim(python_version="3.12")
    # TODO: once LTX2 in diffusers release, change from main to that version (https://github.com/huggingface/diffusers/releases)
    .uv_pip_install(
        "accelerate==1.6.0",
        "av==12.0.0",
        "https://github.com/huggingface/diffusers/archive/refs/pull/12934/head.zip",
        "huggingface-hub==0.36.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.5.1",
        "peft==0.17.0",
        "sentencepiece==0.2.0",
        "torch==2.7.0",
        "transformers==4.51.3",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App(APP_NAME)

VOLUME_NAME = "ltx2-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")
MODEL_VOLUME_NAME = "ltx2-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

MODEL_PATH = Path("/models")
image = image.env({"HF_HOME": str(MODEL_PATH)})


MINUTES = 60  # seconds

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
        from diffusers import LTX2LatentUpsamplePipeline, LTX2Pipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        import torch


        self.pipe = LTX2Pipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )

        self.pipe.to("cuda")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            MODEL_ID, subfolder="latent_upsampler", torch_dtype=torch.bfloat16
        )
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipe.vae, latent_upsampler=latent_upsampler
        )
        self.upsample_pipe.to(device="cuda", dtype=torch.bfloat16)

        scale = getattr(self.upsample_pipe.latent_upsampler.config, "rational_spatial_scale", None)
        print(f"🧠 LTX2: latent upsampler rational_spatial_scale={scale}")

    @modal.method()
    def generate(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=40,
        num_frames=121,
        width=768,
        height=512,
        frame_rate = 24.0,
        guidance_scale=4.0,
        seed=42,
        detailer_lora_scale=1.0,
        stage2_distilled_lora_scale=1.0,
    ):

        import torch

        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.pipe.load_lora_weights(
            "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
            weight_name="ltx-2-19b-ic-lora-detailer.safetensors",
            adapter_name="detailer",
        )
        self.pipe.set_adapters("detailer", adapter_weights=detailer_lora_scale)
        print(f"🧠 LTX2: adapters available (stage1)={self.pipe.get_list_adapters()}")
        print(f"🧠 LTX2: adapters active (stage1)={self.pipe.get_active_adapters()}")
        print("🧠 LTX2: starting base latent generation")
        video_latent, audio_latent = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )
        print(
            "🧠 LTX2: base latent output shape="
            f"{getattr(video_latent, 'shape', None)} dtype={getattr(video_latent, 'dtype', None)}"
        )
        print("🧠 LTX2: starting latent upsampler")
        upsample_start = time.time()
        video_latent = self.upsample_pipe(
            latents=video_latent,
            output_type="latent",
            generator=generator,
            return_dict=False,
        )[0]
        upsample_elapsed = time.time() - upsample_start
        print(
            f"🧠 LTX2: upsampler complete in {upsample_elapsed:.2f}s "
            f"shape={getattr(video_latent, 'shape', None)} dtype={getattr(video_latent, 'dtype', None)}"
        )
        print("🧠 LTX2: starting stage 2 generation")
        audio_latent = audio_latent.to(self.pipe.audio_vae.dtype)
        generated_mel_spectrograms = self.pipe.audio_vae.decode(audio_latent, return_dict=False)[0]
        stage1_audio = self.pipe.vocoder(generated_mel_spectrograms)
        self.pipe.vae.enable_tiling()
        self.pipe.unload_lora_weights()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.load_lora_weights(
            "Lightricks/LTX-2",
            weight_name="ltx-2-19b-distilled-lora-384.safetensors",
            adapter_name="distilled_stage2",
        )
        self.pipe.set_adapters("distilled_stage2", adapter_weights=stage2_distilled_lora_scale)
        print(f"🧠 LTX2: adapters available (stage2)={self.pipe.get_list_adapters()}")
        print(f"🧠 LTX2: adapters active (stage2)={self.pipe.get_active_adapters()}")
        stage2_width = width * 2
        stage2_height = height * 2
        sigma0 = STAGE_2_DISTILLED_SIGMAS[0]
        noise = torch.randn(
            video_latent.shape,
            generator=generator,
            device=video_latent.device,
            dtype=video_latent.dtype,
        )
        video_latent = video_latent + noise * sigma0
        video, _audio_unused = self.pipe(
            latents=video_latent,
            audio_latents=audio_latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=stage2_width,
            height=stage2_height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            sigmas=STAGE_2_DISTILLED_SIGMAS,
            guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_dict=False,
        )
        audio = stage1_audio
        print("🧠 LTX2: stage 2 generation complete")
        video = (video * 255).round().astype("uint8")
        from diffusers.pipelines.ltx2.export_utils import encode_video

        video = torch.from_numpy(video)

        mp4_name = slugify(prompt)
        encode_video(
            video[0],
            fps=frame_rate,
            audio=audio[0].float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,  # should be 24000
            output_path=Path(OUTPUTS_PATH) / mp4_name,
        )

        outputs.commit()
        return mp4_name

@app.local_entrypoint()
def main(
    prompt="An animated polar bear walks into an igloo and says 'I'm home! Who is ready to party?'",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    num_inference_steps: int = 40,  
    guidance_scale: float = 4.0,
    num_frames: int = 121, 
    width: int = 768,
    height: int = 512,
    seed: int = 42,
    detailer_lora_scale: float = 1.0,
    stage2_distilled_lora_scale: float = 1.0,
    ):


    ltx2 = LTX2()

    def run():
        print(f"🎥 Generating a video from the prompt '{prompt}'")
        start = time.time()
        mp4_name = ltx2.generate.remote(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed,
            detailer_lora_scale=detailer_lora_scale,
            stage2_distilled_lora_scale=stage2_distilled_lora_scale,
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
