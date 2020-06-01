FROM ubuntu:bionic as build

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    build-essential

RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip3 install --upgrade pip && \
    /app/.venv/bin/pip3 install --upgrade wheel && \
    /app/.venv/bin/pip3 install --upgrade setuptools

COPY download/ /download/

RUN /app/.venv/bin/pip3 install -f /download 'numpy==1.15.2'

COPY requirements.txt /app/
RUN /app/.venv/bin/pip3 install -f /download -r /app/requirements.txt

COPY models /app/models/

# VCTK multi-speaker model
RUN cd /app/models && \
    cat 20171222_deepvoice3_vctk108_checkpoint_step000300000.pth.gz.part-* \
        > 20171222_deepvoice3_vctk108_checkpoint_step000300000.pth.gz && \
    gunzip 20171222_deepvoice3_vctk108_checkpoint_step000300000.pth.gz

# LJSpeech single speaker model
RUN cd /app/models && \
    cat 20180505_deepvoice3_checkpoint_step000640000.pth.gz.part-* \
        > 20180505_deepvoice3_checkpoint_step000640000.pth.gz && \
    gunzip 20180505_deepvoice3_checkpoint_step000640000.pth.gz

# -----------------------------------------------------------------------------

FROM ubuntu:bionic

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    python3 libpython3.6

COPY --from=build /app/.venv/ /app/.venv/
COPY --from=build /app/models/ /app/models/

COPY nltk_data/ /app/.venv/nltk_data/
COPY presets/ /app/presets/
COPY deepvoice3_pytorch /app/deepvoice3_pytorch/
COPY *.py /app/

EXPOSE 5000

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/app/.venv/bin/flask", "run", "--host", "0.0.0.0"]