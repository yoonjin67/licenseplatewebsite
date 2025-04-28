import torch
import coremltools as ct

# Load YOLOv5 model (without post-processing)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.eval()

# Use only the raw model (no postprocessing)
core_model = model.model
core_model.eval()

# Example input
example_input = torch.rand(1, 3, 640, 640)

# Run a forward pass to check output shape
with torch.no_grad():
    out = core_model(example_input)
    print(f"Model output shape: {[o.shape for o in out] if isinstance(out, list) else out.shape}")

# Trace the model
traced = torch.jit.trace(core_model, example_input)

# Convert to CoreML (.mlmodel) without 'mlprogram'
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="images", shape=example_input.shape, scale=1/255.0)],
    minimum_deployment_target=ct.target.iOS13,  # <= 핵심! iOS13 이하 타겟 설정
    compute_units=ct.ComputeUnit.CPU_ONLY  # Safe setting for Windows/Linux
)

# Save as .mlmodel
mlmodel.save("yolov5.mlmodel")

print("✅ 변환 완료: yolov5.mlmodel")

