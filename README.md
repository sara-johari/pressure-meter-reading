# pressure-meter-reading
Agent to accurately read pressure meter readings.

pressure-meter-agent/
│
├── README.md
├── .gitignore
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                    # Original unprocessed meter images
│   ├── processed/              # Output from preprocessing pipeline
│   └── ground_truth.csv        # Image filename + actual reading + unit
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Orchestrates all steps in order
│   │   ├── denoise.py          # Bilateral filter / fastNlMeans
│   │   ├── perspective.py      # Ellipse detection + warp correction
│   │   ├── crop.py             # Hough circle detection + tight crop
│   │   ├── contrast.py         # CLAHE + gamma correction
│   │   ├── pointer.py          # Canny overlay + HSV channel boost
│   │   └── scale.py            # Adaptive threshold blend for tick marks
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── reader.py           # Calls VLM API with processed image
│   │   ├── prompt.py           # Prompt templates for the VLM
│   │   └── parser.py           # Extracts structured reading from VLM output
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── compare.py          # Compares VLM readings vs ground truth
│       ├── metrics.py          # MAE, MAPE, accuracy within tolerance
│       └── report.py           # Generates summary CSV + visual report
│
├── scripts/
│   ├── run_preprocessing.py    # CLI: process all images in data/raw/
│   ├── run_agent.py            # CLI: run VLM on all processed images
│   └── run_evaluation.py       # CLI: compare predictions vs ground truth
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_agent.py
│   └── test_evaluation.py
│
└── outputs/
    ├── predictions.csv         # VLM output per image
    └── evaluation_report.csv   # Final comparison table