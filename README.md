# Implementation for Mobile Device Connection to the P2PFL Network

This project is part of the Bachelor's Thesis (TFG) titled:  
**"Integration of a Decentralized Federated Learning Library in Mobile Devices"**.

This branch contains the specific development required to enable mobile devices to connect to the decentralized federated learning (**P2PFL**) network. The implementations here are fundamental for establishing communication between mobile nodes and the proxy, enabling collaborative training.

---

## Project Description

Decentralized federated learning is an innovative approach that allows multiple devices to collaborate on training machine learning models without sharing their local data. This approach is particularly relevant for mobile devices, where privacy and efficient data usage are crucial.

In this branch, the following implementations have been developed:

1. Configuration and adjustments to the **proxy node** to enable connections with mobile clients.
2. Adaptation of the network architecture to support multiple devices.
3. Implementation of TensorFlow and its conversion mechanisms to TFLite.
4. Implementation of a new aggregation mechanism.
---

## Authorship

This work has been carried out by **Bosco Suárez-LLanos Outeiriño** as part of the Bachelor's Thesis (TFG) at **UDC/Facultad de Ingeniería Informática**.  
If you have questions, concerns, or feedback about this project, feel free to reach out.

---

## How to Use This Project

To try this project, refer to the README of the main project. To clone with the main project use:

```bash
git clone --recurse-submodules https://github.com/BoscoSO/p2pfl.git
```
or execute this to update it:

```bash
git submodule update --init --recursive
```



---

## Contact

**Author**: [Bosco Suárez-LLanos Outeiriño]  
**Email**: [boscosuarezllo@gmail.com]  
**Main Project Repository**: [Link to Repository](https://github.com/BoscoSO/Mobile_p2pfl)




![GitHub Logo](https://raw.githubusercontent.com/pguijas/p2pfl/main/other/logo.png)

# P2PFL - Federated Learning over P2P networks

[![GitHub license](https://img.shields.io/github/license/pguijas/federated_learning_p2p)](https://github.com/pguijas/p2pfl/blob/main/LICENSE.md)
[![GitHub issues](https://img.shields.io/github/issues/pguijas/federated_learning_p2p)](https://github.com/pguijas/p2pfl/issues)
![GitHub contributors](https://img.shields.io/github/contributors/pguijas/federated_learning_p2p)
![GitHub forks](https://img.shields.io/github/forks/pguijas/federated_learning_p2p)
![GitHub stars](https://img.shields.io/github/stars/pguijas/federated_learning_p2p)
![GitHub activity](https://img.shields.io/github/commit-activity/m/pguijas/federated_learning_p2p)
[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpguijas%2Fp2pfl%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pguijas/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://join.slack.com/t/p2pfl/shared_invite/zt-2lbqvfeqt-FkutD1LCZ86yK5tP3Duztw)

P2PFL is a general-purpose open-source library for the execution (simulated and in real environments) of Decentralized Federated Learning systems, specifically making use of P2P networks and the Gossip protocol.

> **PROXY CONSIDERATIONS**: Just a POC, edge nodes actually act as workers, in a real scenario, they would act as real nodes.

## ✨ Key Features

P2PFL offers a range of features designed to make decentralized federated learning accessible and efficient. For detailed information, please refer to our [documentation](https://pguijas.github.io/p2pfl/).

| Feature          | Description                                      |
|-------------------|--------------------------------------------------|
| 🚀 Easy to Use   | Get started quickly with our intuitive API.       |
| 🛡️ Reliable     | Built for fault tolerance and resilience.       |
| 🌐 Scalable      | Leverages the power of peer-to-peer networks.    |
| 🧪 Versatile     | Experiment in simulated or real-world environments.|
| 🔒 Private       | Prioritizes data privacy with decentralized architecture.|
| 🧩 Flexible      | Integrate with PyTorch and TensorFlow (coming soon!).|
| 📈 Real-time Monitoring | Manage and track experiment through [P2PFL Web Services](https://p2pfl.com). | 
| 🧠 Model Agnostic | Use any machine learning model you prefer (e.g., PyTorch models). |
| 📡 Communication Protocol Agnostic | Choose the communication protocol that best suits your needs (e.g., gRPC). |

## 🔌 Integrations

> todo

- Hugging Face Datasets
- PyTorch
- Tensorflow

## 📥 Installation

> **Note:** We recommend using Python 3.9 or lower. We have found some compatibility issues with Python 3.10 and PyTorch.

### 👨🏼‍💻 For Users

```bash
pip install "p2pfl[torch]"
```

### 👨🏼‍🔧 For Developers

#### 🐍 Python (using Poetry)

```bash
git clone https://github.com/pguijas/p2pfl.git
cd p2pfl
poetry install -E torch 
```

> **Note:** Use the extras (`-E`) flag to install specific dependencies (e.g., `-E torch`). Use `--no-dev` to exclude development dependencies.

#### 🐳 Docker

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```

## 🎬 Quickstart

To start using P2PFL, follow our [quickstart guide](https://pguijas.github.io/p2pfl/quickstart.html) in the documentation.

## 📚 Documentation & Resources

* **Documentation:** [https://pguijas.github.io/p2pfl/](https://pguijas.github.io/p2pfl)
* **End-of-Degree Project Report:** [other/memoria.pdf](other/memoria.pdf)
* **Open Source Project Award Report:** [other/memoria-open-source.pdf](other/memoria-open-source.pdf)

## 🤝 Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines. Please adhere to the project's code of conduct in `CODE_OF_CONDUCT.md`.

## 💬 Community

Connect with us and stay updated:

* [**GitHub Discussions:**](https://github.com/pguijas/p2pfl/discussions) - For general discussions, questions, and ideas.
* [**GitHub Issues:**](https://github.com/pguijas/p2pfl/issues) - For reporting bugs and requesting features.
* [**Google Group:**](https://groups.google.com/g/p2pfl) - For discussions and announcements.
* [**Slack:**](https://join.slack.com/t/p2pfl/shared_invite/zt-2lbqvfeqt-FkutD1LCZ86yK5tP3Duztw) - For real-time conversations and support.


## ⭐ Star History

A big thank you to the community for your interest in P2PFL! We appreciate your support and contributions.

[![Star History Chart](https://api.star-history.com/svg?repos=pguijas/p2pfl&type=Date)](https://star-history.com/#pguijas/p2pfl&Date)

## 📜 License

[GNU General Public License, Version 3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)
