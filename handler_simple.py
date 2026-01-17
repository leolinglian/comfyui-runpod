import runpod
import json
import os
import sys
import subprocess
import time
import requests
import base64
from typing import Dict, Optional

COMFYUI_PATH = "/comfyui"
COMFYUI_URL = "http://127.0.0.1:8188"
VOLUME_PATH = "/runpod-volume"
comfyui_process = None

def setup_volume_links():
    if os.path.exists(VOLUME_PATH):
        print("Setting up volume links...")
        for dir_name in ["checkpoints", "loras", "vae", "embeddings"]:
            src = os.path.join(VOLUME_PATH, "models", dir_name)
            dst = os.path.join(COMFYUI_PATH, "models", dir_name)
            if os.path.exists(src) and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.symlink(src, dst)
                print(f"  ✓ Linked: {dir_name}")

def start_comfyui():
    global comfyui_process
    if comfyui_process is not None:
        return
    print("="*60)
    print("Starting ComfyUI...")
    print("="*60)
    setup_volume_links()
    comfyui_process = subprocess.Popen(
        ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--highvram", "--dont-print-server"],
        cwd=COMFYUI_PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    for i in range(40):
        try:
            if requests.get(f"{COMFYUI_URL}/system_stats", timeout=2).status_code == 200:
                print(f"✓ ComfyUI started ({i+1}s)")
                return
        except:
            pass
        time.sleep(1)
    raise Exception("Failed to start ComfyUI")

def create_workflow(prompt: str, negative_prompt: str, width: int = 832, height: int = 1216, seed: Optional[int] = None, steps: int = 8, cfg: float = 1.5) -> Dict:
    if seed is None:
        seed = int(time.time() * 1000000) % 2147483647
    lightning_lora = f"sdxl_lightning_{steps}step_lora.safetensors"
    return {
        "3": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1.0, "model": ["11", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"},
        "4": {"inputs": {"ckpt_name": "RealVisXL_V5.0.safetensors"}, "class_type": "CheckpointLoaderSimple"},
        "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "6": {"inputs": {"text": prompt, "clip": ["11", 1]}, "class_type": "CLIPTextEncode"},
        "7": {"inputs": {"text": negative_prompt, "clip": ["11", 1]}, "class_type": "CLIPTextEncode"},
        "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": "character", "images": ["8", 0]}, "class_type": "SaveImage"},
        "10": {"inputs": {"lora_name": lightning_lora, "strength_model": 1.0, "strength_clip": 1.0, "model": ["4", 0], "clip": ["4", 1]}, "class_type": "LoraLoader"},
        "11": {"inputs": {"lora_name": "Concept_Art_XL.safetensors", "strength_model": 0.7, "strength_clip": 0.7, "model": ["10", 0], "clip": ["10", 1]}, "class_type": "LoraLoader"}
    }

def submit_workflow(workflow: Dict) -> str:
    return requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=10).json()["prompt_id"]

def get_history(prompt_id: str) -> Dict:
    return requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5).json()

def wait_for_completion(prompt_id: str, timeout: int = 120) -> Dict:
    start_time = time.time()
    while time.time() - start_time < timeout:
        history = get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.5)
    raise Exception(f"Timeout: {prompt_id}")

def extract_images(history_data: Dict) -> list:
    images = []
    for node_id, node_output in history_data.get("outputs", {}).items():
        if "images" in node_output:
            for img_info in node_output["images"]:
                img_path = os.path.join(COMFYUI_PATH, img_info.get("type", "output"), img_info.get("subfolder", ""), img_info["filename"]) if img_info.get("subfolder") else os.path.join(COMFYUI_PATH, img_info.get("type", "output"), img_info["filename"])
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        images.append({"filename": img_info["filename"], "data": base64.b64encode(f.read()).decode('utf-8')})
    return images

def handler(event: Dict) -> Dict:
    start_time = time.time()
    try:
        input_data = event.get("input", {})
        user_prompt = input_data.get("prompt")
        if not user_prompt:
            return {"error": "No prompt provided"}
        full_prompt = f"character design, professional illustration, concept art, {user_prompt}, detailed facial features, clean linework, soft cel shading, simple background, high quality, masterpiece"
        negative_prompt = input_data.get("negative_prompt", "photograph, photo, realistic, photorealistic, ugly, deformed, bad anatomy, blurry, low quality, nsfw, nude")
        width, height, seed = input_data.get("width", 832), input_data.get("height", 1216), input_data.get("seed")
        mode = input_data.get("mode", "balanced")
        config = {"fast": {"steps": 4, "cfg": 1.0}, "balanced": {"steps": 8, "cfg": 1.5}, "quality": {"steps": 25, "cfg": 7.0}}.get(mode, {"steps": 8, "cfg": 1.5})
        workflow = create_workflow(full_prompt, negative_prompt, width, height, seed, steps=config["steps"], cfg=config["cfg"])
        print(f"Generating ({mode}): {user_prompt[:60]}...")
        prompt_id = submit_workflow(workflow)
        history_data = wait_for_completion(prompt_id, timeout=120)
        images = extract_images(history_data)
        if not images:
            return {"error": "No images generated"}
        elapsed = time.time() - start_time
        print(f"✓ Generated in {elapsed:.2f}s")
        return {"status": "success", "prompt_id": prompt_id, "images": images, "generation_time": elapsed, "metadata": {"prompt": user_prompt, "width": width, "height": height, "seed": workflow["3"]["inputs"]["seed"], "steps": config["steps"], "cfg": config["cfg"], "mode": mode}}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "generation_time": time.time() - start_time}

print("="*60)
print("  Fast Character Generator")
print("  RealVisXL V5.0 + Lightning")
print("="*60)
try:
    start_comfyui()
    print("\nWarming up...")
    submit_workflow(create_workflow("test", "ugly", 512, 512, 99999, 4, 1.0))
    print("✓ Ready!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

runpod.serverless.start({"handler": handler})
