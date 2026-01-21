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
modal run ltx2_modal.py --prompt "Style: Pixar-style 3D animation, EXT. snowy Arctic dusk at the mouth of a rounded ice igloo, a wide establishing shot with soft blue rim light and warm amber spill from inside as a friendly polar bear with big expressive eyes, a teal knit scarf, and fluffy fur pads across crunchy snow toward the entrance. The bear’s shoulders sway with a cheerful gait, breath puffing in the cold, while gentle wind and soft footsteps on snow mix with a cozy crackle from inside. The camera slowly dollies closer as the bear ducks under the ice arch and steps into the glowing interior, textured with smooth ice walls and a woven rug. In a medium shot, the bear straightens, smiles wide, and gestures with a paw, mouth moving clearly to the words, 'i’m home! what’s for dinner? i hope it’s salmon!' in a warm, upbeat voice. The bear’s ears perk and eyes sparkle as it looks around expectantly, and the camera eases to a gentle stop on a welcoming, intimate framing, pixar style acting and timing."
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
