import litserve as ls
from litserve.specs.openai import ChatCompletionRequest, ChatMessage
from vllm import LLM, SamplingParams


class PixtralAPI(ls.LitAPI):
    def setup(self, device):
        model_name = "mistralai/Pixtral-12B-2409"
        max_img_per_msg = 5

        self.llm = LLM(
            model=model_name,
            tokenizer_mode="mistral",
            limit_mm_per_prompt={"image": max_img_per_msg},
            max_model_len=32768,
            enable_chunked_prefill=False,
        )

    def decode_request(self, request: ChatCompletionRequest):
        temperature = request.temperature or 0.7
        max_tokens = request.max_tokens or 2048
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]
        return sampling_params, messages

    def predict(self, model_inputs):
        sampling_params, messages = model_inputs
        yield self.llm.chat(messages, sampling_params)

    def encode_response(self, outputs):
        for out in outputs:
            content = out[0].outputs[0].text
            yield ChatMessage(role="assistant", content=content)


if __name__ == "__main__":
    api = PixtralAPI()
    server = ls.LitServer(api, spec=ls.OpenAISpec())
    server.run(port=8000)
