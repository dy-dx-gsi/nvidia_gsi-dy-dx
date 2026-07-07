# OmniForge: Complete Environment Setup Guide

OmniForge is an enterprise-grade Sim-to-Real orchestration pipeline. To evaluate the complete live execution loop, your host environment requires specific configurations across cloud APIs, synthetic data generation, and robotics middleware.

## 1. NVIDIA NIM (Cloud LLM Orchestration)

OmniForge uses NVIDIA's NIM endpoints for highly optimized reasoning using Llama-3.1-8B-Instruct.

**Setup Steps:**
1. Navigate to the [NVIDIA Build API Platform](https://build.nvidia.com/).
2. Create an account or log in.
3. Select any Llama 3.1 model and click **"Get API Key"**.
4. In the root directory of this project, create a file named `.env`.
5. Add your key to the file:
   ```env
   NVIDIA_API_KEY=nvapi-your_api_key_here
   ```

## 2. Python Environment & Dependencies

**Setup Steps:**
1. Create a Python virtual environment (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   ```bash
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 3. Local Vision Edge Training (Ultralytics YOLOv8)

To train the YOLOv8 model locally at high speed, your system must have CUDA available.

**Setup Steps:**
1. Install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
2. Install [cuDNN](https://developer.nvidia.com/cudnn).
3. Ensure PyTorch is installed with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

> **Note**: If CUDA is not available, the system will automatically fall back to CPU training.

## 4. Synthetic Data Generation

OmniForge includes a **Universal Procedural Synthetic Data Engine** built with OpenCV + NumPy that generates realistic defect images for any part/defect combination — no additional installation needed.

For production-grade photorealistic rendering, OmniForge also supports **NVIDIA Omniverse Replicator** (Isaac Sim). This requires the full Isaac Sim application runtime, which can be installed from the [NVIDIA Isaac Sim GitHub](https://github.com/isaac-sim/IsaacSim).

## 5. ROS 2 (Sim-to-Real Deployment)

The `ros2_bridge` node broadcasts parameters to physical factory robots using the standard robotics middleware (`rclpy`).

**Setup Steps (Windows via WSL2):**
ROS 2 on native Windows is possible but often problematic. The industry standard is to use Ubuntu via WSL2.
1. Install WSL2 on Windows: `wsl --install`
2. Open your Ubuntu terminal.
3. Install **ROS 2 Jazzy** (or Humble) by following the [official Ubuntu documentation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
4. Run the ROS 2 script from within WSL2:
   ```bash
   source /opt/ros/jazzy/setup.bash
   python3 ros2_bridge/deploy_node.py
   ```

> **Note**: If ROS 2 is not installed, the deployment step will simulate the broadcast output so the pipeline remains functional.

---
*Note: Due to the complexity of requiring three disparate environments (Cloud API, Omniverse Runtime, ROS 2 Middleware) simultaneously, the OmniForge Streamlit Dashboard (`app.py`) is designed to trigger these scripts as isolated subprocesses and gracefully handle cases where a specific runtime is not available.*
