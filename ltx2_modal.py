import string
import time
from pathlib import Path

import modal

MODEL_ID = "Lightricks/LTX-2"
APP_NAME = "ltx2-text-to-video"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # TODO: once LTX2 in diffusers release, change from main to that version (https://github.com/huggingface/diffusers/releases)
    .uv_pip_install(
        "accelerate==1.6.0",
        "av==12.0.0",
        "https://github.com/huggingface/diffusers/archive/refs/heads/main.zip",
        "huggingface-hub==0.36.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.5.1",
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
        from diffusers import LTX2Pipeline
        import torch


        self.pipe = LTX2Pipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

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
    ):

        video, audio = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="np",
            return_dict=False,
        )
        video = (video * 255).round().astype("uint8")
        import torch
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
    guidance_scale: float = 4.5,
    num_frames: int = 121, 
    width: int = 768,
    height: int = 512,
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
        )
        duration = time.time() - start
        print(f"🎥 Client received video in {int(duration)}s")
        print(f"🎥 LTX2 video saved to Modal Volume at {mp4_name}")

        local_dir = Path("/tmp/ltx2")
        local_dir.mkdir(exist_ok=True, parents=True)
        local_path = local_dir / mp4_name
        local_path.write_bytes(b"".join(outputs.read_file(mp4_name)))
        print(f"🎥 LTX2 video saved locally at {local_path}")

    run()
