import os
import tempfile
import traceback
from collections import OrderedDict
from typing import List, Dict, Any, Optional
import multiprocessing
import torch 
from PIL import Image # For _fake_boxes and _fake_mask dimensions


# =========================================================================
# === CRITICAL: CUSTOMIZE THESE THREE LINES TO MATCH YOUR TRAINING CONFIG ===
# =========================================================================
# 1. Your segmentation model encoder (e.g., "resnet34", "efficientnet-b3", "resnet50")
SEG_ENCODER_NAME = "resnet34" 

# 2. The number of classes you trained for (e.g., 1 for binary, 2 if background was class 0)
SEG_NUM_CLASSES = 1 

# 3. Your SegFormer backbone name (e.g., "nvidia/mit-b0", "nvidia/mit-b3", "nvidia/mit-b5")
SEGFORMER_MODEL_HF = "nvidia/mit-b0"
# =========================================================================


# ---------------------------
# Worker used by multiprocessing (top-level function, imports inside)
# ---------------------------
def _torch_load_worker(in_path: str, out_path: str, use_weights_only: bool, conn):
    """
    Worker to load a torch object on CPU and write it out so parent can remap to device.
    Sends ("ok", None) or ("err", traceback_str) through conn.
    Heavy imports (torch) are performed lazily here so the parent import remains light.
    """
    try:
        # Lazy import of torch to avoid doing it during module import in the parent.
        import torch # type: ignore

        if use_weights_only:
            try:
                # Attempt weights_only=True first, if available
                loaded = torch.load(in_path, map_location="cpu", weights_only=True)
            except TypeError:
                # Fallback for older torch versions
                loaded = torch.load(in_path, map_location="cpu")
        else:
            loaded = torch.load(in_path, map_location="cpu")

        torch.save(loaded, out_path)
        conn.send(("ok", None))
    except Exception:
        tb = traceback.format_exc()
        try:
            conn.send(("err", tb))
        except Exception:
            # fallback: write the traceback to a file next to out_path
            try:
                with open(out_path + ".err.txt", "w", encoding="utf-8") as f:
                    f.write(tb)
            except Exception:
                pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
# ---------------------------
# Helper: robust torch.load wrapper
# ---------------------------
def _robust_torch_load(path: str, device: str, timeout: int = 120):
    """
    Wrapper that tries to load using a worker (weights_only path) then falls back to direct loads.
    device is a string like "cpu" or "cuda:0".
    Heavy import of torch is delayed until needed.
    """
    tmp_out = None
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    try:
        fd, tmp_out = tempfile.mkstemp(suffix=".pth")
        os.close(fd)

        proc = multiprocessing.Process(
            target=_torch_load_worker, args=(path, tmp_out, True, child_conn)
        )
        proc.start()
        proc.join(timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(1)
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            raise TimeoutError(f"torch.load worker timed out after {timeout} seconds (weights_only path).")

        # read status
        try:
            if parent_conn.poll(0.1):
                status, payload = parent_conn.recv()
            else:
                status, payload = ("err", "No status returned from worker")
        except Exception:
            status, payload = ("err", "Failed to receive status from worker")

        if status == "ok":
            # lazy import and load using requested device mapping
            import torch # type: ignore
            loaded = torch.load(tmp_out, map_location=device)
            try:
                os.remove(tmp_out)
            except Exception:
                pass
            return loaded
        else:
            try:
                os.remove(tmp_out)
            except Exception:
                pass
            # Fall through to manual direct load attempts below if worker failed
            raise RuntimeError(f"Worker error while loading {path}:\n{payload}")

    except Exception as first_err:
        # cleanup
        try:
            if tmp_out and os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass

        # fallback: try direct torch.load with lazy import
        try:
            import torch # type: ignore
            return torch.load(path, map_location=device)
        except Exception as second_err:
            try:
                import torch # type: ignore
                return torch.load(path, map_location="cpu")
            except Exception as final_err:
                raise RuntimeError(
                    "All attempts to torch.load() failed.\n"
                    f"Worker attempt error: {repr(first_err)}\n"
                    f"Parent attempt error: {repr(second_err)}\n"
                    f"Final CPU attempt error: {repr(final_err)}"
                ) from final_err
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass


# ---------------------------
# SegmentationModels class
# ---------------------------
class SegmentationModels:
    """
    Loads segmentation models and provides `.segment(image_path, model_name)` that returns a PIL.Image.
    Heavy libraries (torch, numpy, PIL, smp) are imported inside methods to keep module import lightweight.
    """

    def __init__(self, device: str = "cpu"):
        self.device_str = device
        self.device = None
        self.models = {}
        base = os.path.dirname(__file__)
        self.model_paths = {
            "segformer": os.path.join(base, "welding_segformer_best.pth"),
            "unet": os.path.join(base, "unet.pth"),
            "mcnn": os.path.join(base, "mcnn_trained_best.pth"),       # <-- swapped in MCNN here
        }
        self.load_timeout_seconds = 120
        
        try:
            self._load_models()
        except Exception as e:
            print(f"[SegmentationModels] warning during initial load: {e}")

    def _ensure_torch_device(self):
        if self.device is None:
            import torch # type: ignore
            self.device = torch.device(self.device_str)

    def _build_model(self, name: str):
        """
        Build architecture for state_dict loading, ensuring only local weights are used.
        """
        lname = name.lower()
        
        # --- UNet, UNet++ and MCNN (using UNet as placeholder for MCNN) ---
        if "unet" in lname or "mcnn" in lname:
            try:
                import segmentation_models_pytorch as smp # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "segmentation_models_pytorch is required to build Unet/Unet++ architectures. "
                    "Install with: pip install segmentation-models-pytorch"
                ) from e
            
            # --- Using variables defined globally (set by the user at the top of the file) ---
            encoder_name = SEG_ENCODER_NAME 
            num_classes = SEG_NUM_CLASSES 
            # ----------------------------------------------------------------------------------

            if "unet++" in lname or "unetpp" in lname:
                return smp.UnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_weights=None,   # Forces model to use only your state_dict (FIX)
                    classes=num_classes, 
                    activation=None
                )
            else:
                if "mcnn" in lname:
                    print(f"[SegmentationModels] Building architecture for '{name}' using UNet as a structural placeholder.")
                
                return smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,   # Forces model to use only your state_dict (FIX)
                    classes=num_classes, 
                    activation=None
                )

        # --- SegFormer ---
        if "segformer" in lname:
            try:
                # Lazy import HuggingFace libraries
                from transformers import SegformerForSemanticSegmentation, SegformerConfig
            except ImportError:
                raise RuntimeError(
                    "SegFormer requires 'transformers' library. "
                    "Install with: pip install transformers accelerate"
                )
            
            # --- Using variables defined globally (set by the user at the top of the file) ---
            model_name_hf = SEGFORMER_MODEL_HF 
            num_labels = SEG_NUM_CLASSES 
            # ----------------------------------------------------------------------------------

            try:
                # Load configuration to build the structure, relying on a cached/local config.
                config = SegformerConfig.from_pretrained(
                    model_name_hf, 
                    num_labels=num_labels, 
                    local_files_only=False 
                )
                
                # Build the model architecture (no weights loaded yet)
                model = SegformerForSemanticSegmentation(config)
                return model
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to build SegFormer architecture for '{model_name_hf}': {e}. "
                    "Ensure the model name, number of labels, and transformers dependencies are correct."
                )

        raise ValueError(f"Unknown model name '{name}' - update _build_model() accordingly.")

    def _load_models(self):
        """
        Attempts to load all model files. This calls _robust_torch_load which will spawn a worker if needed.
        """
        import torch # type: ignore 

        for name, path in self.model_paths.items():
            if not os.path.exists(path):
                print(f"[SegmentationModels] file not found: {path} (skipping '{name}')")
                continue
            print(f"[SegmentationModels] Loading '{name}' from {path} ...")
            try:
                # Load the state dict using the robust loader
                loaded = _robust_torch_load(path, device=self.device_str, timeout=self.load_timeout_seconds)
            except Exception as e:
                # IMPORTANT: Print detailed error for load failure
                print(f"[SegmentationModels] Failed to load '{path}' into memory: {e}")
                continue

            if isinstance(loaded, (dict, OrderedDict)):
                try:
                    # 1. Build the empty architecture
                    model = self._build_model(name)
                except Exception as e:
                    print(f"[SegmentationModels] Could not build model '{name}': {e}")
                    continue

                sd = loaded
                # 2. Fix `module.` prefix if the model was saved with DataParallel
                if any(k.startswith("module.") for k in sd.keys()):
                    new_sd = OrderedDict()
                    for k, v in sd.items():
                        new_sd[k.replace("module.", "")] = v
                    sd = new_sd

                try:
                    # 3. Load the state dict into the empty architecture
                    model.load_state_dict(sd)
                    model.to(self.device_str)
                    model.eval()
                    self.models[name] = model
                    print(f"[SegmentationModels] Loaded state_dict into '{name}' successfully.")
                except Exception as e:
                    # IMPORTANT: Print detailed error for structure mismatch
                    print(f"[SegmentationModels] Failed to load state_dict into model '{name}' (Structure Mismatch/Key Error): {e}")
                    print(f"Hint: Check the global SEG_ENCODER_NAME ({SEG_ENCODER_NAME}) and SEG_NUM_CLASSES ({SEG_NUM_CLASSES}) are correct for this model.")
                    continue
            else:
                try:
                    # Handle case where the full model object (not just state dict) was saved
                    model = loaded
                    model.to(self.device_str)
                    model.eval()
                    self.models[name] = model
                    print(f"[SegmentationModels] Loaded full model object for '{name}'.")
                except Exception as e:
                    print(f"[SegmentationModels] Loaded object from '{path}' but could not treat as model: {e}")
                    continue

        if not self.models:
            print("[SegmentationModels] Warning: no models loaded successfully. UI will use simulated outputs.")

    def get(self, name: str):
        return self.models.get(name)

    # ---------------------------
    # Inference helper: segment
    # ---------------------------
    def segment(self, image_path: str, model_name: str = "segformer") -> Dict[str, Any]: # Return Dict for status
        """
        Runs segmentation and returns a dictionary containing the PIL.Image mask and the model name used.
        """
        # lazy imports
        from PIL import Image, ImageDraw # type: ignore
        import numpy as np # type: ignore
        import torch # type: ignore
        
        actual_model_name = model_name
        model = self.get(model_name)
        
        # --- Fallback Logic (Modified) ---
        if model is None:
            if self.models:
                # Fallback to the first available model
                fallback_model_name = next(iter(self.models.keys()))
                model = self.get(fallback_model_name)
                print(f"[SegmentationModels] requested '{model_name}' not found. Falling back to '{fallback_model_name}'.")
                actual_model_name = fallback_model_name
            else:
                print(f"[SegmentationModels] No models available (requested: {model_name}) — returning simulated mask.")
                return {"mask_img": self._fake_mask(image_path), "model_name_used": "simulated"}

        # Ensure torch device is ready
        try:
            self._ensure_torch_device()
        except Exception:
            return {"mask_img": self._fake_mask(image_path), "model_name_used": "simulated"}

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[SegmentationModels] Failed to open image '{image_path}': {e}. Returning fake mask.")
            return {"mask_img": self._fake_mask(image_path), "model_name_used": "simulated"}

        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                out = model(tensor)
            except Exception:
                try:
                    out = model.forward(tensor)
                except Exception:
                    print(f"[SegmentationModels] Model '{actual_model_name}' forward failed. Returning fake mask.")
                    return {"mask_img": self._fake_mask(image_path), "model_name_used": "simulated"}

        # Postprocess
        try:
            out_t = out
            if isinstance(out_t, dict):
                for k in ("out", "logits", "pred", "mask", "masks"):
                    if k in out_t:
                        out_t = out_t[k]
                        break
                else:
                    out_t = next(iter(out_t.values()))

            if isinstance(out_t, (tuple, list)):
                out_t = out_t[0]

            out_t = out_t.detach().cpu()

            if out_t.ndim == 4:
                n, c, h, w = out_t.shape
                if c == 1:
                    prob = torch.sigmoid(out_t)[0, 0]
                    mask_np = (prob.numpy() * 255).astype(np.uint8)
                else:
                    pred = torch.argmax(out_t, dim=1)[0].numpy().astype(np.uint8)
                    mask_np = (pred > 0).astype(np.uint8) * 255
            elif out_t.ndim == 3:
                if out_t.shape[0] == 1:
                    prob = torch.sigmoid(out_t[0])
                    mask_np = (prob.numpy() * 255).astype(np.uint8)
                else:
                    c = out_t.shape[0]
                    if c == 1:
                        prob = torch.sigmoid(out_t[0])
                        mask_np = (prob.numpy() * 255).astype(np.uint8)
                    else:
                        pred = torch.argmax(out_t, dim=0).numpy().astype(np.uint8)
                        mask_np = (pred > 0).astype(np.uint8) * 255
            else:
                print(f"[SegmentationModels] Unexpected output tensor shape: {out_t.shape}. Returning fake mask.")
                return {"mask_img": self._fake_mask(image_path), "model_name_used": "simulated"}
        except Exception as e:
            print(f"[SegmentationModels] Failed to postprocess model output: {e}. Returning fake mask.")
            return {"mask_img": self._fake_mask(image_path), "model_name_used": "simulated"}

        try:
            mask_img = Image.fromarray(mask_np).convert("L")
            if mask_img.size != img.size:
                mask_img = mask_img.resize(img.size, resample=Image.NEAREST)
            return {"mask_img": mask_img.convert("RGB"), "model_name_used": actual_model_name}
        except Exception as e:
            print(f"[SegmentationModels] Error converting mask to image: {e}. Returning fake mask.")
            return {"mask_img": self._fake_mask(image_path), "model_name_used": "simulated"}

    def _fake_mask(self, image_path: str):
        # lazy import PIL
        from PIL import Image, ImageDraw # type: ignore
        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
        except Exception:
            w, h = 640, 480
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        padw = max(10, w // 6)
        padh = max(10, h // 6)
        draw.rectangle([padw, padh, w - padw, h - padh], fill=255)
        return mask.convert("RGB")


# ---------------------------
# DetectionModels class
# ---------------------------
class DetectionModels:
    """
    Loads detection models (YOLO) and provides `.detect(image_path, model_name)` that returns bounding boxes.
    Heavy libraries (ultralytics, torch) are imported inside methods to keep module import lightweight.
    """

    def __init__(self, device: str = "cpu"):
        self.device_str = device
        self.models = {}
        base = os.path.dirname(__file__)
        self.model_paths = {
            "yolo11m": os.path.join(base, "best.pt"), # Assumed final filename
            # Add other detection models here if needed
        }
        
        try:
            self._load_models()
        except Exception as e:
            print(f"[DetectionModels] warning during initial load: {e}")

    def _load_models(self):
        """
        Loads models using ultralytics YOLO class.
        """
        # Lazy import ultralytics
        try:
            from ultralytics import YOLO # type: ignore
        except Exception as e:
            print(f"[DetectionModels] ultralytics library not found. Install with: pip install ultralytics")
            return

        for name, path in self.model_paths.items():
            if not os.path.exists(path):
                print(f"[DetectionModels] file not found: {path} (skipping '{name}')")
                continue
            print(f"[DetectionModels] Loading '{name}' from {path} ...")
            try:
                # YOLO constructor handles loading and device mapping automatically
                model = YOLO(path)
                self.models[name] = model
            except Exception as e:
                print(f"[DetectionModels] Failed to load YOLO model '{path}': {e}")
                continue

        if not self.models:
            print("[DetectionModels] Warning: no detection models loaded successfully.")

    def get(self, name: str):
        return self.models.get(name)

    def detect(self, image_path: str, model_name: str = "yolo11m") -> List[Dict[str, Any]]:
        """
        Runs detection and returns a list of bounding box dictionaries.
        """
        model = self.get(model_name)
        if model is None:
            if self.models:
                model = next(iter(self.models.values()))
                print(f"[DetectionModels] requested '{model_name}' not found, falling back to available model.")
            else:
                print("[DetectionModels] No detection models available — returning simulated boxes.")
                return _fake_boxes(image_path)

        try:
            # Predict uses the device specified in __init__
            results = model.predict(image_path, device=self.device_str, verbose=False)
        except Exception as e:
            print(f"[DetectionModels] Model predict failed: {e}. Returning fake boxes.")
            return _fake_boxes(image_path)

        boxes: List[Dict[str, Any]] = []
        color_map = {} # To assign a consistent color per class
        default_colors = ["#FF4D4D", "#4DA6FF", "#FFC700", "#7CFC00", "#FF69B4"]
        color_idx = 0

        for result in results:
            for box in result.boxes:
                # x1, y1, x2, y2 
                coords = box.xyxy[0].tolist() 
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                # Get class name using the loaded model's names dictionary
                if model.names is not None and cls_id in model.names:
                    cls_name = model.names[cls_id]
                else:
                    cls_name = f"class_{cls_id}"

                # Assign color
                if cls_name not in color_map:
                    color_map[cls_name] = default_colors[color_idx % len(default_colors)]
                    color_idx += 1

                boxes.append({
                    "class": cls_name,
                    "confidence": f"{conf:.2f}",
                    # Coordinates are already in pixel values [x1, y1, x2, y2]
                    "coords": [int(c) for c in coords], 
                    "color": color_map[cls_name]
                })

        if not boxes:
            return _fake_boxes(image_path)
            
        return boxes


def _fake_boxes(image_path: str):
    """
    Returns simulated bounding boxes if model loading or inference fails.
    """
    try:
        w_img, h_img = Image.open(image_path).size
    except Exception:
        w_img, h_img = 640, 480
        
    return [{
        "class": "simulated_defect",
        "confidence": "0.99",
        "coords": [w_img // 4, h_img // 4, w_img * 3 // 4, h_img * 3 // 4],
        "color": "#FFA500"
    }]

# ---------------------------
# Top-level helper: detect_bboxes (Delegates to DetectionModels)
# ---------------------------
def detect_bboxes(image_path: str, model_name: str = "yolo11m") -> List[Dict[str, Any]]:
    """
    Top-level detection function, delegates to DetectionModels.
    """
    det = DetectionModels(device="cpu") 
    
    # Try the requested model first
    if model_name in det.models:
        return det.detect(image_path, model_name)

    # Fallback to any available model
    if det.models:
        fallback_model = next(iter(det.models.keys()))
        print(f"[detect_bboxes] Requested model '{model_name}' not found. Falling back to '{fallback_model}'.")
        return det.detect(image_path, fallback_model)
        
    print("[detect_bboxes] No detection models loaded. Returning simulated boxes.")
    return _fake_boxes(image_path)


# ---------------------------
# Module smoke test (only runs if executed directly)
# ---------------------------
if __name__ == "__main__":
    print("models.py smoke test - constructing SegmentationModels (this will import torch)...")
    seg = SegmentationModels(device="cpu")
    print("Available segmentation models:", list(seg.models.keys()))
    
    print("\nmodels.py smoke test - constructing DetectionModels (this will import ultralytics)...")
    det = DetectionModels(device="cpu")
    print("Available detection models:", list(det.models.keys()))
    
    # Create a dummy image for testing if none exists
    dummy_img_path = "test_image.png"
    if not os.path.exists(dummy_img_path):
        from PIL import Image
        img = Image.new("RGB", (640, 480), color = 'red')
        img.save(dummy_img_path)
        print(f"Created a dummy image at {dummy_img_path}")
        
    # Test segmentation
    if seg.models:
        try:
            print("Testing segmentation...")
            # Capture the result from the dictionary returned by segment
            result = seg.segment(dummy_img_path, next(iter(seg.models.keys())))
            mask = result["mask_img"]
            used_model = result["model_name_used"]
            print(f"Segmentation successful. Mask size: {mask.size}. Model Used: {used_model}")
        except Exception as e:
            print(f"Segmentation test failed: {e}")
            
    # Test detection
    if det.models:
        try:
            print("Testing detection...")
            boxes = det.detect(dummy_img_path, next(iter(det.models.keys())))
            print(f"Detection successful. Found {len(boxes)} boxes.")
        except Exception as e:
            print(f"Detection test failed: {e}")
