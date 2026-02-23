#!/bin/bash
ZONES="us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c us-west1-a us-west1-b us-west2-b us-west2-c us-west4-a us-west4-b us-central1-b us-central1-c us-central1-f"

for zone in $ZONES; do
  echo "Trying $zone..."
  gcloud compute instances create cs224n-gpu \
    --zone=$zone \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
  if [ $? -eq 0 ]; then
    echo "Success in $zone!"
    break
  fi
done
