import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from peft import get_peft_model, LoraConfig, TaskType


# 모델 정의
class KoBERT_TF_Model(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        base_model = AutoModel.from_pretrained(
            "monologg/kobert", trust_remote_code=True
        )
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=True,
            r=32,
            lora_alpha=128,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        lora_model = get_peft_model(base_model, lora_config)
        lora_model = self.patch_forward(lora_model)
        self.bert = lora_model

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def patch_forward(self, peft_model):
        original_forward = peft_model.model.forward

        def new_forward(*args, **kwargs):
            kwargs.pop("labels", None)
            return original_forward(*args, **kwargs)

        peft_model.model.forward = new_forward
        return peft_model

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = output.last_hidden_state[:, 0, :]
        return self.classifier(cls_vec)


# 모델 및 토크나이저 로드
model_path = hf_hub_download(
    repo_id="yniiiiii/kobert-tf-model-lora-nositu", filename="pytorch_model.bin"
)
model = KoBERT_TF_Model()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
