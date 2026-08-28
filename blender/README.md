# Blender dataset generation

This directory preserves the Blender scene setup and RenderSynth provenance
used to generate the local dataset. The retained
[`material_dataset_master.blend`](material_dataset_master.blend) scene is the
small demonstrator for the scene, camera, lighting, and material-generation
setup. Other `.blend` files and Blender autosave/backup files are intentionally
gitignored; the Python scripts and JSON pipeline configuration are tracked.

The source textures and RenderSynth add-on are external dependencies. They are
not bundled in this repository and the local texture/material directories,
render outputs, and training data remain gitignored. A fresh checkout therefore
needs Blender, RenderSynth, and separately obtained source textures before the
generation scripts can be run.

Run from the repository root, for example:

```text
blender --background blender/material_dataset_master.blend \
  --python blender/scripts/render_full_dataset.py -- --renders 30
```

Generation scripts read split/material manifests from `metadata/` and write
renders to the local `dataset/` directory. Validate Blender and the RenderSynth
add-on in a smoke run before a full render. Do not regenerate or repartition
the locked test split without recording a new dataset version.
