from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.engine.ttrl_trainer import TTRLBackend, TTRLModelAdapter, TTRLRequestStream


class TinyTTRLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 8)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        features = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        logits = self.projection(features)
        loss = logits.mean()
        if labels is not None:
            loss = loss + labels.float().mean() * 0.0
        return SimpleNamespace(loss=loss, logits=logits)


def test_request_stream_reads_jsonl(tmp_path):
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(
        '{"id":"a","prompt":"Question A","input_ids":[1,2],"labels":[1,2]}\n'
        '{"request_id":"b","question":"Question B","input_ids":[3,4],"labels":[3,4]}\n',
        encoding="utf-8",
    )

    records = list(TTRLRequestStream(request_path))

    assert [record["request_id"] for record in records] == ["a", "b"]
    assert records[0]["question"] == "Question A"


def test_request_stream_reads_json_list(tmp_path):
    request_path = tmp_path / "requests.json"
    request_path.write_text(
        '{"requests":[{"question":"Question A","input_ids":[1],"labels":[1]}]}',
        encoding="utf-8",
    )

    records = list(TTRLRequestStream(request_path))

    assert records[0]["request_id"] == "ttrl_request_0"
    assert records[0]["input_ids"] == [1]


def test_model_adapter_supports_prediction_and_training_call():
    adapter = TTRLModelAdapter(TinyTTRLModel())

    prediction = adapter.forward(
        {
            "question": "What is shown?",
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }
    )
    training_output = adapter(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        labels=torch.tensor([[1, 2, 3]]),
    )

    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["logits"] is not None
    assert training_output.loss.requires_grad
    assert training_output.logits.shape[-1] == 8


def test_ttrl_backend_requires_request_stream(tmp_path):
    backend = TTRLBackend({"ttrl": {"output_dir": str(tmp_path)}})

    with pytest.raises(ValueError, match="request_path"):
        backend._build_request_stream()
