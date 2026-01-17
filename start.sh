#!/bin/bash
echo "=========================================="
echo "Starting ComfyUI RunPod Handler"
echo "=========================================="
nvidia-smi
python /comfyui/handler.py
