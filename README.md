# LTX-2 on Modal

This repo deploys the LTX-2 text and image to video two stage pipeline on Modal with proxy-auth-protected API endpoints and a Modal CLI entrypoint. This uses the full 19B dev LTX-2.

## Prerequisites

1. **Create a Modal Account:** Sign up for a free account at [modal.com](https://modal.com).

2. **Install Modal Client:** Install the Modal client library and set up your authentication token.

   ```bash
   pip install modal
   modal setup
   ```

3. **Clone this Repository:**

   ```bash
   git clone https://github.com/Square-Zero-Labs/modal-ltx2
   cd modal-ltx2
   ```

4. Agree to the Gemma terms here https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized.

5. Create a Hugging Face token with read permissions on https://huggingface.co/ so you can pull the Gemma weights.

6. Place the Hugging Face token in an .env file that looks like .env.example

7. Save your Hugging Face token as a Modal secret:

```bash
modal secret create HF_TOKEN HF_TOKEN="$(python - <<'PY'
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() == "HF_TOKEN":
        print(value.strip().strip('"').strip("'"))
        break
PY
)"
```

## Deploy the app:

```bash
modal deploy ltx2_modal.py
```

Once deployed, Modal prints the web endpoint URL; set it in `.env` as `LTX2_API_URL` (the base URL; the API uses `/generate`).

## Generate a video locally via Modal

Use the local entrypoint to kick off a run on Modal and save the result:

Output resolution is exactly the `--width`/`--height` you pass. Defaults are `1536x1024` (two-stage LTX-2 defaults). Width and height must be divisible by 64. For 16:9, we saw best results using `1280x704`.

```bash
modal run ltx2_modal.py --width 1280 --height 704 --seconds 10 --prompt "INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker’s face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises. Baker (whispering dramatically): 'Today… I achieve perfection.' He leans even closer, nose nearly touching the glass. 'Golden edges. Soft center. The gods themselves will smell these cookies and weep.' pixar style acting and timing"
```

Enable the detailer LoRA (applies to both stages; default is off):

```bash
modal run ltx2_modal.py --width 1280 --height 704 --seconds 10 --use-detailer-lora --prompt "Style: cinematic-realistic, INT. upscale sleek corner office with warm daylight, polished wood, and soft reflections on metal fixtures. A beautiful blonde coworker in a tailored blazer and smart trousers strides into the office with brisk heel clicks while a beautiful redhead coworker in a fitted blouse looks up from her monitor at a minimalist desk. The blonde slows near the desk, excited smile spreading as she says, 'I figured out how to create LTX-2 videos fast and for free!' The redhead straightens in her chair, eyes bright, and responds with a delighted laugh, 'No way!' The camera pushes in to a two-shot as each raises a hand and complete a crisp high five."
```

Image-to-video example:

```bash
modal run ltx2_modal.py --width 1280 --height 704 --seconds 5 --image-path "./inputs/panda-at-work.jpg" --image-strength 0.9 --prompt 'A pixar style panda in an office waves his paw in greeting. He says in a warm, cheerful voice: Have a seat! I would be happy to help you generate videos with LTX-2!'
```

## Generate a video via the API

Create a Proxy Auth Token in the Modal UI for this workspace, then load the API URL and token values from your `.env` file:

```bash
export $(grep -E '^(LTX2_API_URL|LTX2_PROXY_TOKEN_ID|LTX2_PROXY_TOKEN_SECRET)=' .env | xargs)
```

Kick off a generation job (captures the Modal `job_id`):

```bash
JOB_ID=$(curl -sS -X POST \
  -H "Modal-Key: $LTX2_PROXY_TOKEN_ID" \
  -H "Modal-Secret: $LTX2_PROXY_TOKEN_SECRET" \
  -F 'prompt=Style: Pixar-style 3D animation, EXT. snowy Arctic dusk at the mouth of a rounded ice igloo...' \
  -F "seconds=5" \
  -F "width=1280" \
  -F "height=704" \
  "$LTX2_API_URL/generate" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
)
echo "JOB_ID=$JOB_ID"
```

Check status with a HEAD request (returns `202` while running, `200` when ready):

```bash
curl -I \
  -H "Modal-Key: $LTX2_PROXY_TOKEN_ID" \
  -H "Modal-Secret: $LTX2_PROXY_TOKEN_SECRET" \
  "$LTX2_API_URL/generate/$JOB_ID"
```

Download the video bytes when ready (save locally to `outputs/`):

```bash
mkdir -p outputs
curl -L \
  -H "Modal-Key: $LTX2_PROXY_TOKEN_ID" \
  -H "Modal-Secret: $LTX2_PROXY_TOKEN_SECRET" \
  "$LTX2_API_URL/generate/$JOB_ID" \
  -o "outputs/${JOB_ID}.mp4"
```

Image-to-video via the API (send an image file path):

```bash
JOB_ID=$(curl -sS -X POST \
  -H "Modal-Key: $LTX2_PROXY_TOKEN_ID" \
  -H "Modal-Secret: $LTX2_PROXY_TOKEN_SECRET" \
  -F 'prompt=A pixar style panda in an office waves his paw in greeting. He says in a warm, friendly voice: Have a seat! I would be happy to help you generate videos with LTX-2!' \
  -F "seconds=6" \
  -F "width=1280" \
  -F "height=704" \
  -F "image_path=@inputs/panda-at-work.jpg" \
  -F "image_strength=0.9" \
  "$LTX2_API_URL/generate" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
)
echo "JOB_ID=$JOB_ID"
```

Image-to-video via the API (send an image URL):

```bash
JOB_ID=$(curl -sS -X POST \
  -H "Modal-Key: $LTX2_PROXY_TOKEN_ID" \
  -H "Modal-Secret: $LTX2_PROXY_TOKEN_SECRET" \
  -F 'prompt=A pixar style panda in an office waves his paw in greeting. He says in a warm, upbeat voice: Have a seat! I would be happy to help you generate videos with LTX-2!' \
  -F "seconds=6" \
  -F "width=1280" \
  -F "height=704" \
  -F "image_url=https://example.com/panda-at-work.jpg" \
  -F "image_strength=0.9" \
  "$LTX2_API_URL/generate" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
)
echo "JOB_ID=$JOB_ID"
```

## Development Notes

### Git Subtree Management

When originally added:

```bash
git subtree add --prefix LTX-2 https://github.com/Lightricks/LTX-2 main --squash
```

If the original `LTX-2` repository is updated and you want to incorporate those changes into this project, you can pull the updates using the following command:

```bash
git subtree pull --prefix LTX-2 https://github.com/Lightricks/LTX-2 main --squash
```
