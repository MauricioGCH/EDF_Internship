import ray
import torch

def test_gpu():
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("No GPUs available")

ray.init()
ray.get(ray.remote(test_gpu).remote())