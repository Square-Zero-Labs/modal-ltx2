# LTX-2 on Modal

This repo deploys the LTX-2 text-to-video pipeline on Modal with a token-protected API and a Modal CLI entrypoint.

## Modal setup

Create a Modal secret that holds the API token:

```bash
modal secret create ltx2-api-token LTX2_API_TOKEN=your-token-here
```

Create a Modal secret for Hugging Face (Gemma is gated):

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

Deploy the app:

```bash
modal deploy ltx2_modal.py
```

Once deployed, Modal prints the web endpoint URL. Export it for API calls:

```bash
export LTX2_API_URL=https://<your-modal-url>
export LTX2_API_TOKEN=your-token-here
```

## Generate a video locally via Modal

Use the local entrypoint to kick off a run on Modal and save the result:

```bash
modal run ltx2_modal.py --seconds 10 --prompt "INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker’s face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises. Baker (whispering dramatically): 'Today… I achieve perfection.' He leans even closer, nose nearly touching the glass. 'Golden edges. Soft center. The gods themselves will smell these cookies and weep.' pixar style acting and timing"
```

## Call the API directly

```bash
curl -X POST "$LTX2_API_URL" \
  -H "Authorization: Bearer $LTX2_API_TOKEN" \
  -H "Content-Type: application/json" \
  -o ltx2-output.mp4 \
  -d '{"prompt":"a cinematic mountain sunrise"}'
```

## API schema

`POST` JSON payload:

```json
{
  "prompt": "a cinematic mountain sunrise",
  "negative_prompt": null,
  "height": 576,
  "width": 1024,
  "num_frames": 48,
  "fps": 24,
  "guidance_scale": 3.0,
  "num_inference_steps": 30,
  "seed": 0
}
```

Pass the token via `Authorization: Bearer <token>` (or `X-API-Token`).
