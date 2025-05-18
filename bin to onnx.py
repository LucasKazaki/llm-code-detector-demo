import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Point to your saved student model directory
student_dir = r"student_model"  # <-- replace with your actual folder
output_path = "model.onnx"

# 2. Load the student model
model = AutoModelForSequenceClassification.from_pretrained(student_dir)
model.eval()

# 3. (Optional) load tokenizer just to confirm vocab
tokenizer = AutoTokenizer.from_pretrained(student_dir)

# 4. Create a dummy input (batch_size=1, seq_len=512)
#    Here we use token ID 0 for simplicity; adjust if your tokenizer uses different pad token
dummy_input = torch.zeros((1, 512), dtype=torch.int64)

# 5. Export to ONNX
torch.onnx.export(
    model,
    (dummy_input,),
    output_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":    {0: "batch_size", 1: "sequence_length"},
        "logits":       {0: "batch_size"}
    },
    opset_version=13
)

print(f"Exported ONNX model to: {output_path}")
