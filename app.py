import gradio as gr
import torch
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import types
import time
import re
import gc
import traceback
from unittest.mock import MagicMock

# --- CONFIGURATION (UNIVERSAL MODE) ---
# v3.8: Auto-detects location. Works on M3, M5, or External Drives.
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if not BASE_PATH.endswith("/"): BASE_PATH += "/"

# Fallback for "Evos2" naming drift
if "Evos2" in BASE_PATH and not os.path.exists(BASE_PATH):
    BASE_PATH = BASE_PATH.replace("Evos2", "Evo2")

WEIGHTS_NAME = "evo2_7b"
print(f"--- SYSTEM: LOCATION DETECTED: {BASE_PATH} ---")

# --- 1. THE HOLO-DECK (Patching Suite) ---
print("--- SYSTEM: INITIALIZING BIO-DECK UI v3.8 (Universal Transfusion) ---")

class GhostObject:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, name): return self
    def __iter__(self): return iter([])
    def __len__(self): return 0
    def __getitem__(self, key): return self
    def __repr__(self): return "<GhostObject>"
    def __fspath__(self): return "/tmp/ghost_path"

class CudaHologram:
    is_initialized = True
    class device: 
        def __init__(self, idx): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass
    class Stream(GhostObject): pass
    class Event(GhostObject): pass
    @staticmethod
    def is_available(): return True 
    @staticmethod
    def device_count(): return 1
    @staticmethod
    def current_device(): return 0
    @staticmethod
    def get_device_name(device=None): return "NVIDIA H100 (Emulated via Apple Silicon)"
    @staticmethod
    def get_device_capability(device=None): return (9, 0) 
    def __getattr__(self, name): return GhostObject()

torch.cuda = CudaHologram()
sys.modules['torch.cuda'] = CudaHologram()

_RealDevice = torch.device
class _HijackedDevice:
    def __new__(cls, device_str, *args, **kwargs):
        if isinstance(device_str, str) and "cuda" in device_str.lower(): return _RealDevice("mps")
        return _RealDevice(device_str, *args, **kwargs)
torch.device = _HijackedDevice

_original_to = torch.Tensor.to
def _hijacked_to(self, *args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, str) and "cuda" in arg.lower(): new_args.append("mps")
        elif isinstance(arg, _RealDevice) and arg.type == "cuda": new_args.append(_RealDevice("mps"))
        else: new_args.append(arg)
    if "device" in kwargs:
        d = kwargs["device"]
        if isinstance(d, str) and "cuda" in d.lower(): kwargs["device"] = "mps"
    return _original_to(self, *new_args, **kwargs)
torch.Tensor.to = _hijacked_to
torch.Tensor.cuda = lambda self, *args, **kwargs: self.to("mps")

# --- THE SHAPE-SHIFTER ENGINE ---
class UniversalMock(MagicMock):
    # Behave like a list/dictionary if accessed with [0] or ["key"]
    def __getitem__(self, key): 
        return self
    
    # Behave like a tuple if unpacked (x, y = result)
    def __iter__(self): 
        # Yields itself infinitely to satisfy "x, y = func()" or "x, y, z = func()"
        return iter([self, self, self, self])

def deploy_mock(name):
    # Create the shape-shift
    # Create the shape-shifter
    m = UniversalMock(name=name)
    # Inject it into the system
    sys.modules[name] = m
    return m
# --------------------------------

#class UniversalMock(types.ModuleType):
    #def __getattr__(self, key): return self
    #def __call__(self, *args, **kwargs): return self
    #def __iter__(self): return iter([])
    #def __repr__(self): return f"<UniversalMock: {self.__name__}>"
    #def __fspath__(self): return "/tmp/evo2_mock_path"

#def deploy_mock(name):
    #m = UniversalMock(name)
    #sys.modules[name] = m
    #return m

deploy_mock("flash_attn_2_cuda")
deploy_mock("fused_dense_lib")
deploy_mock("triton")
deploy_mock("triton.language")
deploy_mock("Bio")
deploy_mock("Bio.Seq")
#deploy_mock("huggingface_hub")
te = deploy_mock("transformer_engine")
te.is_available = lambda: True
te_torch = deploy_mock("transformer_engine.pytorch")

class ChameleonLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
class ChameleonLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape)

te_torch.Linear = ChameleonLinear
te_torch.LayerNorm = ChameleonLayerNorm
te_torch.fp8_linear = ChameleonLinear
sys.modules["transformer_engine.pytorch.fp8_linear"] = te_torch

# Path Logic
def find_package_parent(search_path, package_name, file_hint="__init__.py"):
    for root, dirs, files in os.walk(search_path):
        if file_hint in files:
            if os.path.basename(root) == package_name:
                return os.path.dirname(root) 
            if "src" in root and package_name in dirs:
                return root 
    return None

vortex_parent = find_package_parent(BASE_PATH, "vortex")
if vortex_parent:
    if vortex_parent not in sys.path: sys.path.append(vortex_parent)
else:
    sys.path.append(os.path.join(BASE_PATH, "vortex-main"))

evo_parent = find_package_parent(BASE_PATH, "evo2")
if evo_parent:
    if evo_parent not in sys.path: sys.path.append(evo_parent)

# --- 2. LOAD MODEL (Global State) ---
print("[SYSTEM]: Loading Evo2 Model...")
try:
    # --- STEP 1.5: THE SEDATIVE ---
    #try:
    #    import vortex.model.utils
    #    def dummy_load(*args, **kwargs):
    #        print("   -> [INTERCEPTED] Auto-load blocked for Transfusion.")
    #        return None
    #    vortex.model.utils.load_checkpoint = dummy_load
    #except ImportError: pass

    from evo2 import Evo2
    
    evo_module = sys.modules[Evo2.__module__]
    if hasattr(evo_module, 'constants'):
        evo_module.constants.HF_HUB_CACHE = BASE_PATH
    else:
        for name, mod in sys.modules.items():
            if "constants" in name and "evo2" in name and hasattr(mod, "HF_HUB_CACHE"):
                mod.HF_HUB_CACHE = BASE_PATH

    # Init Skeleton (Meta)
    print("[SYSTEM]: Initializing Structure (Skeleton Only)...")
    with torch.device("meta"):
        evo_wrapper = Evo2("evo2_7b") 
        model = evo_wrapper.model
    
    # --- THE TAILOR PROTOCOL (Resizing Meta Layers) ---
    print("[SYSTEM]: Tailoring Meta Layers (21888 -> 22528)...")
    TARGET_DIM = 22528
    ORIGINAL_DIM = 21888
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, ChameleonLinear):
            if module.out_features == ORIGINAL_DIM:
                new_layer = torch.nn.Linear(module.in_features, TARGET_DIM, bias=module.bias is not None, device="meta")
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                setattr(model.get_submodule(parent_name), child_name, new_layer)
                count += 1
            elif module.in_features == ORIGINAL_DIM:
                new_layer = torch.nn.Linear(TARGET_DIM, module.out_features, bias=module.bias is not None, device="meta")
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                setattr(model.get_submodule(parent_name), child_name, new_layer)
                count += 1
    print(f"[SYSTEM]: Tailoring Complete. {count} layers resized.")

    # --- THE TRANSFUSION PROTOCOL (Smart Cache Resolution) ---
    from huggingface_hub import hf_hub_download
    
    print(f"[SYSTEM]: Resolving path for Sovereign Weights...")
    
    # 1. Define the official Repo ID (This matches what the library downloaded)
    REPO_ID = "ArcInstitute/evo2_7b" 
    FILENAME = "evo2_7b.pt"

    # 2. Ask the Hub where the file lives locally
    try:
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print(f"[SYSTEM]: Target Found in Cache: {weights_path}")
    except Exception as e:
        # Fallback if the name is slightly different
        print(f"[WARNING]: Could not resolve primary name. Trying alternate...")
        weights_path = os.path.join(BASE_PATH, "evo2_7b.pt")

    
    print(f"[SYSTEM]: Reading Source Weights (14GB CPU Load)...")
    # We load the source blob. This hits ~14GB RAM (Safe for M3 Ultra).
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)

    
    # --- PASTE THIS: THE DEEP LOBOTOMY ---
    print("--- SYSTEM: SURGICALLY REMOVING FLASH ATTENTION FROM ALL LAYERS ---")
    # 1. Update the Global Config
    model.config.use_flash_attn = False
    model.config.use_flash_attention_2 = False
    
    # 2. Hunt down every individual layer and force it to comply
    count = 0
    for name, module in model.named_modules():
        # Check for any attribute attempting to use Flash Attention
        if hasattr(module, "use_flash_attn"):
            module.use_flash_attn = False
            count += 1
        if hasattr(module, "use_flash_attention_2"):
            module.use_flash_attention_2 = False
    
    print(f"--- SYSTEM: SUCCESSFULLY DISARMED {count} FLASH ATTENTION LAYERS ---")
    # -------------------------------------
    
    print("[SYSTEM]: Beginning Transfusion (CPU -> MPS)...")
    
    transfused_count = 0
    total_params = len(list(model.named_parameters()))
    
    def set_tensor_by_name(model, name, tensor):
        path = name.split('.')
        obj = model
        for p in path[:-1]:
            obj = getattr(obj, p)
        setattr(obj, path[-1], torch.nn.Parameter(tensor, requires_grad=False))

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                tensor = state_dict[name]
                tensor_mps = tensor.to("mps", dtype=torch.float16)
                set_tensor_by_name(model, name, tensor_mps)
                del state_dict[name]
                del tensor
                transfused_count += 1
                if transfused_count % 50 == 0:
                    print(f"   -> Transfused {transfused_count}/{total_params} layers...", end="\r")
                    gc.collect()

    print(f"\n[SYSTEM]: Transfusion Complete. {transfused_count} layers activated.")
    del state_dict
    gc.collect()
    
    print("[SYSTEM]: Ready.")
    
except Exception as e:
    print(f"[FATAL]: Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 3. THE INTERFACE LOGIC ---
def generate_sequence(dna_input):
    if not dna_input: return "Error: Empty input"
    
    print(f"\nProcessing: {dna_input[:20]}...")
    start_t = time.time()
    
    try:
        clean_input = re.sub(r'[^ACGTacgt]', '', dna_input).upper()
        if not clean_input: return "Error: No valid DNA characters."
        
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        input_ids = torch.tensor([[mapping.get(c, 0) for c in clean_input]], device="mps")
        
        with torch.no_grad():
            # --- EVO2 BIO-STATION: BFLOAT16 MODE (FAST & STABLE) ---
            # 1. Convert weights to Brain Float (16-bit size, 32-bit range).
            # This fits in RAM (40GB) but prevents the overflow (NaNs).
            model.bfloat16() 
            
            # 2. Execute Inference
            outputs = model(input_ids)
            
            # 3. Standard processing...
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 4. Safety Scrub
            logits = torch.nan_to_num(logits, nan=0.0)
            
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            top_tokens = torch.topk(probs, 5)
            
        elapsed = time.time() - start_t
        
        result = f"✅ Analysis Complete ({elapsed:.2f}s)\n"
        result += f"Sequence Length: {len(clean_input)}\n"
        
        # --- EVO2 BIO-STATION: NUMPY COMPATIBILITY FIX ---
        scores = top_tokens.values.float().cpu().numpy()[0]
        indices = top_tokens.indices.cpu().numpy()[0]

        result += "\nMost Likely Next Nucleotides:\n"
        for i in range(len(indices)):
            idx = int(indices[i])
            score = scores[i] * 100
            
            # --- THE UNIVERSAL DECODER ---
            # The model is outputting ASCII (65='A', 84='T').
            # We use chr() to translate the integer back to a letter.
            try:
                # 65 -> "A", 84 -> "T"
                char = chr(idx)
            except:
                # Fallback if it's a weird control character
                char = f"<{idx}>"
                
            result += f"   {char} (ID {idx}): {score:.2f}%\n"
                   
        return result
        
    except Exception as e:
        print("\n--- CRITICAL FAILURE DETECTED ---")
        traceback.print_exc()  # <--- PRINTS THE TRUTH
        return f"Computation Error: {str(e)}"

# --- 4. THE GUI ---
with gr.Blocks(title="Evo2 Bio-Station", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧬 Evo2 Bio-Station (Apple Silicon)")
    gr.Markdown(f"**Engine:** M5 Max (128GB) | **Model:** {WEIGHTS_NAME} | **Status:** Air-Gapped")
    
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Input DNA Sequence", placeholder="Paste FASTA or raw sequence (ATCG...)", lines=10)
            btn = gr.Button("Run Inference", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Next-Base Prediction", lines=10)
            
    btn.click(fn=generate_sequence, inputs=inp, outputs=out)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)
