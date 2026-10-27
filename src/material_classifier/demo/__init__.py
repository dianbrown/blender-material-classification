"""Local Material Studio inference. Importing this package never loads ML libraries."""

CLASS_NAMES = ("wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather")
MODEL_NAMES = {
    "frozen_selected": "Frozen DINOv2 + ConvNeXt",
    "frozen_control": "Frozen ConvNeXt control",
    "resnet_fixed": "ResNet-50 baseline",
    "dinov2_finetuned": "Fine-tuned DINOv2",
}
LOCK_PATH = "reports/results/v3/final_model_lock.json"


class DemoError(Exception):
    """A safe, actionable protocol error; full diagnostics stay on stderr."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
