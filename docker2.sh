#! /bin/bash
docker run --rm --ipc=host --runtime=nvidia -v ~/development/mjalea/data:/data -v ~/development/mjalea/votenet:/workspace -it eslexoro/scanx:pytorch
