# LTX-2 on Modal

This repo deploys the LTX-2 text-to-video pipeline on Modal with a token-protected API and a Modal CLI entrypoint.

## Modal setup

Create a Modal secret that holds the API token:

```bash
modal secret create ltx2-api-token LTX2_API_TOKEN=your-token-here
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
modal run ltx2_modal.py --prompt "a cinematic mountain sunrise"
```

## Call the API directly

```bash
curl -X POST "$LTX2_API_URL" \
  -H "Authorization: Bearer $LTX2_API_TOKEN" \
  -H "Content-Type: application/json" \
  -o ltx2-output.mp4 \
  -d '{"prompt":"a cinematic mountain sunrise","height":576,"width":1024,"num_frames":48,"fps":24,"guidance_scale":3.0,"num_inference_steps":30,"seed":0}'
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
