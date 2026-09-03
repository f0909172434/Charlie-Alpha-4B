from charlie_alpha.data_pipeline import _chat_length


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [1, 2, 3, 4]


def test_chat_length_requests_token_ids_not_batch_encoding() -> None:
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert _chat_length(FakeTokenizer(), messages) == 4
