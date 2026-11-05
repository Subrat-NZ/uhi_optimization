# UHI RunPod Serverless worker: wind model + T model + RH model in one image. Build from the REPO ROOT:
#   docker build --platform linux/amd64 -f worker/Dockerfile -t jagathmcd/uhi-rht-worker:v2 .
# (without Docker: worker/build_push_crane.sh builds the same image)
# All checkpoints are baked in (wind ~660 MB, T + RH ~270 MB each): no network volume, any datacenter works.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    APP_ROOT=/app

WORKDIR /app
COPY worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY scripts/ /app/scripts/
COPY pressure_calculator/ /app/pressure_calculator/
COPY T_model/ /app/T_model/
COPY RH_model/ /app/RH_model/
COPY wind_api/model/__init__.py wind_api/model/infrenace.py /app/wind_api/model/
COPY wind_api/model/turbdiff/ /app/wind_api/model/turbdiff/
COPY wind_api/model/model/ /app/wind_api/model/model/
COPY app/__init__.py app/runpod_client.py app/wind_ai.py /app/app/
COPY worker/handler.py /app/worker/handler.py

# fail the build (not the first job) if a checkpoint is missing or is a Git LFS pointer
RUN for f in /app/T_model/latest_net_G.pth /app/RH_model/*_net_G.pth /app/wind_api/model/model/best.ckpt; do \
      test "$(stat -c %s "$f")" -gt 1000000 || { echo "bad checkpoint: $f"; exit 1; }; done

CMD ["python", "-u", "/app/worker/handler.py"]
