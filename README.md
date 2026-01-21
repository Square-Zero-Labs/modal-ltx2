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

```bash
modal run ltx2_modal.py --seconds 10 --prompt "INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly fogged glass door. Warm golden light glows around freshly baked cookies. The baker’s face fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Subtle reflections move across the glass as steam rises. Baker (whispering dramatically): 'Today… I achieve perfection.' He leans even closer, nose nearly touching the glass. 'Golden edges. Soft center. The gods themselves will smell these cookies and weep.' pixar style acting and timing"
```

Enable the detailer LoRA (applies to both stages; default is off):

```bash
modal run ltx2_modal.py --seconds 10 --use-detailer-lora --prompt "Style: cinematic-realistic, INT. upscale sleek corner office with warm daylight, polished wood, and soft reflections on metal fixtures. A beautiful blonde coworker in a tailored blazer and smart trousers strides into the office with brisk heel clicks while a beautiful redhead coworker in a fitted blouse looks up from her monitor at a minimalist desk. The blonde slows near the desk, excited smile spreading as she says, 'I figured out how to create LTX-2 videos fast and for free!' The redhead straightens in her chair, eyes bright, and responds with a delighted laugh, 'No way!' The camera pushes in to a two-shot as each raises a hand and complete a crisp high five."
```

```bash
modal run ltx2_modal.py --seconds 10 --prompt "EXT., DAY claymation medium close-up with sunlight, hand-crafted textures, and visible stop-motion fingerprints; a clay bunny rabbit sits on a tree trunk facing the camera. The rabbit has large cute eyes, white fur and pink ears. His fur is clean and soft looking. A garden is behind the rabbit. The rabbit's glossy clay eyes blink as it leans forward, mouth shaping each word clearly, and it lifts one paw in an excited gesture. 'I've done it! I've figured out how to create LTX-2 videos fast and for free!' while its ears flop with each syllable. The camera slowly pushes in to emphasize the rabbit's face and the tiny creases in the clay as it looks directly into the camera and smiles. Faint nature sounds and soft clay squeaks accompany the stop-motion movement, with the rabbit's voice crisp and centered."
```

```bash
modal run ltx2_modal.py --seconds 10 --prompt "A stunning, determined woman bursts out of a massive, fiery explosion in a collapsing building at golden hour, her hair and clothes whipping in the wind, clutching a gleaming ancient artifact above her head. The camera follows her in a dramatic dynamic push-in from a wide angle, dust and sparks flying around, lighting reflecting off the object's polished surface. She turns toward the camera with joy and relief, yelling with exhilaration: I've figured out how to create LTX-2 videos fast and for free!  \n Emphasize cinematic motion blur, intense warm lighting and synchronized sound of explosion and her voice, ultra-realistic action adventure style"
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
