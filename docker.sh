#! /bin/bash
#docker run --rm --ipc=host --runtime=nvidia -v /workspace/mjalea/data:/data -v ~/development/mjalea/votenet:/workspace -it mjalea/scanx:votenet
docker run --rm --ipc=host --runtime=nvidia -p 25864:25864 -v ~/development/mjalea/data:/storage/data -v ~/development/mjalea/votenet:/workspace -v ~/development/mjalea/artifacts:/artifacts -it imntl/scanx:votenet

