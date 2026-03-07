#!/bin/bash
ZONES="us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east4-a us-east4-c us-west1-a us-west1-b us-west2-a us-west2-b europe-west4-a europe-west4-b europe-west4-c"
for zone in $ZONES; do
  echo "Trying $zone with L4..."
  gcloud compute instances create cs224n-gpu \
    --zone=$zone \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
  if [ $? -eq 0 ]; then
    echo "Success in $zone!"
    break
  fi
done