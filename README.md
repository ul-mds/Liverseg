# FedSeg: Federated Learning for Liver and Tumor Segmentation

FedSeg is a federated learning framework built on **Flower (FLWR)** for
distributed medical image segmentation.\
The project aims to train a collaborative **liver and tumor segmentation
model** across multiple institutions (clients) **without sharing raw
patient data**, improving privacy and enabling multi-center learning.

FedSeg uses:

-   **Flower (FLWR)** for federated orchestration\
-   **PyTorch / PyTorch Lightning** (optional) for training\
-   **UNet / custom segmentation models**\
-   **Medical imaging datasets** (e.g., LiTS, custom datasets) stored
    locally on each client

------------------------------------------------------------------------

## 🚀 Features

-   **Federated Learning Across Multiple Clients**\
    Each client trains locally on its private dataset; only model
    weights are shared.

-   **Secure and Privacy-Preserving**\
    Client data never leaves the device. Optional DP or secure
    aggregation can be integrated.

-   **Modular Design**\
    Easy to plug in different segmentation models or datasets.

-   **Supports Heterogeneous Clients**\
    Clients can run on different hardware or OS environments.

-   **Configurable Federated Strategies**\
    FedAvg by default, with optional support for FedProx, FedOpt, or
    custom weighting.

------------------------------------------------------------------------

## 📂 Project Structure

    FedSeg/
    ├── server/
    │   ├── server.py
    │   ├── config.py
    ├── client/
    │   ├── client.py
    │   ├── dataset.py
    │   ├── train.py
    │   ├── model.py
    ├── utils/
    │   ├── metrics.py
    │   ├── transforms.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🧠 Model Architecture

FedSeg supports several architectures including:

-   **UNet (default)**\
-   **Attention UNet**\
-   **UNet++**\
-   Custom segmentation models

------------------------------------------------------------------------

## 🗂️ Dataset Requirements

Each client stores its own dataset **locally**, including:

-   CT or MRI liver scans\
-   Liver + tumor segmentation masks

Structure:

    /data/
       images/
       masks/

------------------------------------------------------------------------

## 🖥️ Installation

``` bash
git clone https://github.com/<your-username>/FedSeg.git
cd FedSeg
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ How to Run FedSeg

### 1️⃣ Start the Server

``` bash
python server/server.py
```

### 2️⃣ Start Each Client

On each client machine:

``` bash
python client/client.py
```

------------------------------------------------------------------------

## 📊 Evaluation

Metrics include Dice, IoU, Precision, Recall, and Volume Similarity.

------------------------------------------------------------------------

## ⚙️ Configuration

Modify `config.py` to adjust FL rounds, local epochs, learning rate, and
batch size.

------------------------------------------------------------------------

## 🔒 Privacy & Security (Optional Enhancements)

Compatible with differential privacy, secure aggregation, and gradient
clipping.

------------------------------------------------------------------------

## 📝 Roadmap

-   Add secure aggregation\
-   Add pretrained weights\
-   Add benchmark evaluation\
-   Docker support\
-   Notebook demos

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome via PRs or issues.

------------------------------------------------------------------------

## 📄 License

MIT License
