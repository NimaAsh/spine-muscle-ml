# Data Layout

Most scripts expect the environment variable below to point to the root of the source model repository:

```bash
export OPENSIM_MUSCLE_NN_DATA_ROOT=/path/to/dataverse_files
```

Expected layout:

```text
dataverse_files/
├── Female/
│   └── AgeXXXX/
│       └── *.osim
├── Male/
│   └── AgeXXXX/
│       └── *.osim
└── generic/
    └── Thoracolumbar Model/
        ├── Female_Thoracolumbar_Spine_V1/
        │   └── Female_Thoracolumbar_Spine_Model.osim
        └── Male_Thoracolumbar_Spine_V1/
            └── Thoracolumbar_Spine_With_RibCage.osim
```

Notes:

- `Male/` and `Female/` are scanned recursively for subject-specific `.osim` files.
- `generic/` supplies the male and female template models used for scaling and target construction.
- The public repo does not bundle raw subject files.
- The subject-specific source models are from the Framingham Heart Study dataset:
  [https://doi.org/10.7910/DVN/SJ5MVM](https://doi.org/10.7910/DVN/SJ5MVM)
