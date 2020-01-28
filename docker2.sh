#! /bin/bash
#docker run --rm --ipc=host --runtime=nvidia -v /workspace/mjalea/data:/data -v ~/development/mjalea/votenet:/workspace -it mjalea/scanx:votenet
docker run --rm --ipc=host -p 25864:25864 -v ~/data/:/storage/data -v ~/git/votenet:/workspace -v ~/tmp/artifacts:/artifacts -it imntl/scanx:votenet

