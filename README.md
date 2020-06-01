# DeepVoice3 Web Server with Pre-Trained English Models

Based on https://github.com/r9y9/deepvoice3_pytorch

Includes VCTK multi-speaker and LJSpeech single speaker English models.

Hosts an HTTP server on port 5000 that mimics the MaryTTS `/process` endpoint (`GET` with `?INPUT_TEXT=...&VOICE=...*`).

See `Makefile` and `Dockerfile` for details.
