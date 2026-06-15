# Models

Model weights are **not stored in GitHub** (too large). Download from Hugging Face:

**https://huggingface.co/2vhoc/helmet-detection-traffic**

## Quick download

```bash
bash scripts/download_models.sh
```

## Expected layout after download

```
models/
├── stage1/
│   ├── detect_vehicle.pt
│   ├── detect_vehicle.onnx
│   └── args.yaml
├── stage2/
│   ├── mu_bien_so_stage2.pt
│   ├── mu_bien_so_stage2.onnx
│   └── args.yaml
└── stage3/
    ├── ocr_plate.pt
    ├── ocr_plate.onnx
    ├── ocr_plate_640.onnx
    └── args.yaml
```
