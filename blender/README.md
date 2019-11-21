# TODO
### Prepare Blender Dataset
#### Generate Dataset in Blender with blender_dataset_generator in branch votenet_generator.
1. Checkout the `blender_dataset_generator` in branch `votenet_generator`

    git clone https://gitlab.pnk.cc/mjalea/blender_dataset_generator.git
    git checkout votenet_generator

2. Render all nessesary files. For more information see README.md in `blender_dataset_generator`

    blender -t 0 -b blend/abc.blend --python blender_generator.py

#### Prepare data ...
... by running

    python3 blender_data.py --gen_data