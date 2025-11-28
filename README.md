# FedSeg: Federated Learning for Liver and Tumor Segmentation

FedSeg is a federated learning framework built on **Flower (FLWR)** for
distributed medical image segmentation.\
The project aims to train a collaborative **liver and tumor segmentation
model** across multiple institutions (clients) **without sharing raw
patient data**, improving privacy and enabling multi-center learning.

FedSeg uses:

-   **Flower (FLWR)** for federated orchestration\
-   **PyTorch / PyTorch Lightning** (optional) for training\
-   **Medical imaging datasets** (e.g., LiTS, IRCAD) stored
    locally on each client

------------------------------------------------------------------------

## 🚀 Features

-   **Federated Learning Across Multiple Clients**\
    Each client trains locally on its private dataset; only model
    weights are shared.

-   **Modular Design**\
    Easy to plug in different segmentation models or datasets.


-   **Configurable Federated Strategies**\
    FedAvg by default, with optional support for FedProx, FedOpt, or
    custom weighting [TODO].

------------------------------------------------------------------------

## 📂 Project Structure

    FedSeg/
    ├── ...
    ├── utils/
    │   ├── metrics.py
    │   ├── transforms.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🧠 Model Architecture

FedSeg supports several architectures, including:

-   **UNet**
-   **ResUNet**
-   **Sepnet**

------------------------------------------------------------------------

## 🗂️ Dataset Requirements

Each client stores its own dataset **locally**, including:

-   CT \
-   Liver + tumor segmentation masks


------------------------------------------------------------------------

## 🖥️ Installation

``` bash
git clone git@github.com:ul-mds/Liverseg.git
cd Fedseg
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ How to Run FedSeg

### Start the Server

``` bash
python server.py
```

### Start Each Client

On each client machine:

``` bash
python client.py
```

### Perform federated learning simulation
``` bash
python main.py
```

------------------------------------------------------------------------

## 📊 Evaluation

Metrics include Dice and IoU.

------------------------------------------------------------------------

## ⚙️ Configuration

...

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome via PRs or issues.

------------------------------------------------------------------------

## 📄 License

MIT License
