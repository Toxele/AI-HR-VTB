import torch
import numpy as np
from transformers import Wav2Vec2Processor
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model


class Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        self.pool_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def attention_pool(self, hidden_states, attention_mask):
        attention_weights = self.pool_attention(hidden_states)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided for attention pooling")

        attention_weights = attention_weights + (
                (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)

        return weighted_sum

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]

        if attention_mask is not None:
            input_length = attention_mask.size(1)
            hidden_length = hidden_states.size(1)
            ratio = input_length / hidden_length
            indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
            attention_mask = attention_mask[:, indices]
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        pooled = self.attention_pool(hidden_states, attention_mask)
        logits = self.classifier(pooled)

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

        if labels is not None:
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            l2_lambda = 0.01
            l2_reg = torch.tensor(0., device=logits.device)
            for param in self.classifier.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            probs = torch.sigmoid(logits.detach())
            return {"loss": loss, "logits": probs}

        probs = torch.sigmoid(logits)
        return {"logits": probs}


class SmartTurn:
    def __init__(self, model_name: str = "pipecat-ai/smart-turn-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Wav2Vec2ForEndpointing.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        if torch.backends.mps.is_available():
            self.device = "mps"
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_endpoint(self, audio_array: np.ndarray) -> dict:
        """
        Предсказывает, завершён ли аудиосегмент (закончился ли говорящий) или он продолжается.

        Args:
        audio_array: Массив NumPy, содержащий аудиосэмплы с частотой дискретизации 16 кГц

        Returns:
        Словарь с результатами предсказания:
        - prediction: 1 — если сегмент завершён, 0 — если продолжается
        - probability: Вероятность завершения (выход сигмоиды)
        """
        try:
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                padding="max_length",
                truncation=True,
                max_length=16000 * 16,
                return_attention_mask=True,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probability = outputs["logits"][0].item()
                prediction = 1 if probability > 0.5 else 0

            return {
                "prediction": prediction,
                "probability": probability,
            }

        except Exception as e:
            raise Exception(f"Ошибка предсказания конца реплики: {str(e)}")
