import os
import json
import tempfile
import logging
from uuid import uuid4

logging.basicConfig(level=logging.DEBUG)

_LOGGER = logging.getLogger("deepvoice3")

# vctk or ljspeech
_MODEL = os.getenv("DEEPVOICE3_MODEL", "vctk").lower()

# -----------------------------------------------------------------------------

import hparams
import synthesis
import train
import audio

from train import build_model
from train import restore_parts, load_checkpoint
from synthesis import tts

# -----------------------------------------------------------------------------

# Set up hyperparameters
for dummy, v in [
    ("fmin", 0),
    ("fmax", 0),
    ("rescaling", False),
    ("rescaling_max", 0.999),
    ("allow_clipping_in_normalization", False),
    ("window_ahead", 0),
    ("value_projection", False),
    ("key_projection", False),
    ("window_backward", 0),
]:
    if hparams.hparams.get(dummy) is None:
        hparams.hparams.add_hparam(dummy, v)

if _MODEL == "ljspeech":
    preset = "presets/20180505_deepvoice3_ljspeech.json"
else:
    preset = "presets/deepvoice3_vctk.json"

_LOGGER.debug("Preset: %s" % preset)

with open(preset) as f:
    hparams.hparams.parse_json(f.read())

if _MODEL != "ljspeech":
    # Tell we are using DeepVoice3 multispeaker
    hparams.hparams.builder = "deepvoice3_multispeaker"

# Inject frontend text processor
import synthesis
from deepvoice3_pytorch import frontend

synthesis._frontend = getattr(frontend, "en")
train._frontend = getattr(frontend, "en")

# Load model
if _MODEL == "ljspeech":
    checkpoint_path = "models/20180505_deepvoice3_checkpoint_step000640000.pth"
else:
    checkpoint_path = "models/20171222_deepvoice3_vctk108_checkpoint_step000300000.pth"

_LOGGER.debug("Loading model from %s", checkpoint_path)

model = build_model()
model = load_checkpoint(checkpoint_path, model, None, True)
_LOGGER.info("Model loaded")

# -----------------------------------------------------------------------------

from flask import Flask, request, make_response
from flask_cors import CORS

app = Flask("deepvoice3")
CORS(app)

# ---------------------------------------------------------------------

# Mimic the MaryTTS HTTP API (loosely)
# GET /process
# INPUT_TEXT = text to convert to speech
# VOICE = id of speaker (defaults to 32)
# Some good speaker ids: 32, 26, 14, 8, 3, 7, 12, 25
@app.route("/process")
def process():
    text = request.args["INPUT_TEXT"]
    speaker_id = -1

    if _MODEL != "ljspeech":
        speaker_id = int(request.args.get("VOICE", 32))
        _LOGGER.debug("Processing '%s' with speaker id %s", text, speaker_id)
    else:
        _LOGGER.debug("Processing '%s'", text)

    wav_data = text_to_speech(text, speaker_id)

    response = make_response(wav_data)
    response.headers["Content-Type"] = "audio/wav"
    return response


@app.route("/voices")
def voices():
    if _MODEL == "ljspeech":
        return "default en_US female"

    return "\n".join(f"{i:03d} en_US female" for i in range(0, 108))


# ---------------------------------------------------------------------


def text_to_speech(text, speaker_id=-1):
    kwargs = {}
    if speaker_id >= 0:
        kwargs["speaker_id"] = speaker_id

    waveform, alignment, spectrogram, mel = tts(model, text, fast=False, **kwargs)

    with tempfile.SpooledTemporaryFile() as f:
        audio.save_wav(waveform, f)
        f.seek(0)
        return f.read()
