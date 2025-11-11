import os
import time
from typing import Tuple

import cv2
import numpy as np
import torch


def select_device() -> torch.device:
	"""Return CUDA device if available, else CPU."""
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")



def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
	"""Load model from .pt.

	Load order:
	1) TorchScript (jit)
	2) Ultralytics YOLO
	3) torch.load with weights_only=False (trusted local file)
	"""
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model file not found: {model_path}")

	# 1) TorchScript first (jit)
	model = None
	try:
		model = torch.jit.load(model_path, map_location=device)
	except Exception:
		model = None

	if model is not None:
		return model

	# 2) Try Ultralytics YOLO loader if available
	try:
		from ultralytics import YOLO  # type: ignore
		model = YOLO(model_path)
		return model
	except Exception:
		model = None

	# 3) Fallback: regular torch.load with weights_only=False (PyTorch>=2.6 default changed)
	#    Only do this for trusted local checkpoints.
	try:
		model = torch.load(model_path, map_location=device, weights_only=False)
	except TypeError:
		# Older PyTorch versions do not have weights_only arg
		model = torch.load(model_path, map_location=device)

	# Some torch.load returns (model, extra) tuples
	if isinstance(model, (tuple, list)):
		model = model[0]

	if isinstance(model, torch.nn.Module):
		model.to(device)
		model.eval()
		return model

	# For TorchScript, model is already callable
	return model


def preprocess_frame_bgr(
	frame_bgr: np.ndarray,
	input_size: Tuple[int, int] = (224, 224),
):
	"""Convert OpenCV BGR frame to normalized tensor suitable for classification models.

	Assumes ImageNet normalization. Adjust if your model expects different sizes or stats.
	"""
	# Resize (width, height) in OpenCV
	resized = cv2.resize(frame_bgr, input_size, interpolation=cv2.INTER_LINEAR)
	# BGR -> RGB
	rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
	# To float32 [0,1]
	img = rgb.astype(np.float32) / 255.0
	# Normalize (ImageNet mean/std). Change if your model differs.
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	img = (img - mean) / std
	# HWC -> CHW
	img = np.transpose(img, (0, 1, 2))
	# To tensor with batch dim
	tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
	return tensor


def softmax_logits_to_topk(logits: torch.Tensor, k: int = 1):
	"""Return top-k (probabilities, indices) from raw logits."""
	probs = torch.softmax(logits, dim=1)
	values, indices = torch.topk(probs, k=k, dim=1)
	return values.squeeze(0).detach().cpu().numpy(), indices.squeeze(0).detach().cpu().numpy()


def main():
	model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
	device = select_device()
	model = load_model(model_path, device)

	# Open default webcam
	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	if not cap.isOpened():
		raise RuntimeError("Cannot access the webcam. Ensure it's connected and not in use.")

	# Try to infer expected input size from common models; default 224
	input_size = (224, 224)

	fps_avg = 0.0
	fps_alpha = 0.1  # exponential moving average factor

	print("Press 'q' to quit the window.")
	while True:
		start_time = time.time()
		ret, frame = cap.read()
		if not ret:
			print("Failed to read frame from webcam.")
			break

		# If it's an Ultralytics YOLO model, use its API
		annotated = None
		is_yolo = hasattr(model, "predict") and hasattr(model, "names")
		if is_yolo:
			# Ultralytics expects BGR images fine; returns list[Results]
			results_list = model(frame, verbose=False)
			res = results_list[0]
			annotated = res.plot()
			# Compose a brief prediction string from the first box if exists
			if len(res.boxes) > 0:
				conf = float(res.boxes.conf[0].cpu().numpy())
				cls_idx = int(res.boxes.cls[0].cpu().numpy())
				label = None
				try:
					label = model.names.get(cls_idx) if isinstance(model.names, dict) else model.names[cls_idx]
				except Exception:
					label = str(cls_idx)
				text_pred = f"YOLO: {label} {conf:.2f}"
			else:
				text_pred = "YOLO: no detections"
		else:
			# Classification-like models
			with torch.no_grad():
				inp = preprocess_frame_bgr(frame, input_size=input_size).to(device)
				logits = model(inp)
				if isinstance(logits, (tuple, list)):
					logits = logits[0]
				top_probs, top_indices = softmax_logits_to_topk(logits, k=1)
				pred_idx = int(top_indices[0]) if top_indices.ndim > 0 else int(top_indices)
				prob = float(top_probs[0]) if top_probs.ndim > 0 else float(top_probs)
				text_pred = f"Pred: class {pred_idx} | p={prob:.2f}"

		elapsed = time.time() - start_time
		fps = 1.0 / max(elapsed, 1e-6)
		fps_avg = fps_alpha * fps + (1.0 - fps_alpha) * fps_avg

		# Overlay prediction and FPS on the chosen display frame
		text_fps = f"FPS: {fps_avg:.1f} ({fps:.1f})"
		display_frame = annotated if annotated is not None else frame
		cv2.putText(display_frame, text_pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		cv2.putText(display_frame, text_fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

		cv2.imshow("Webcam Inference", display_frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


