from tqdm.notebook import tqdm
import functools
from lorem_text import lorem
from .utils import num_tokens_from_messages

class CostEstimator:
    MODEL_SYNONYMS = {
        "gpt-4": "gpt-4-0613",
        "gpt-3-turbo": "gpt-3.5-turbo-0125",
        "gpt-4o": "gpt-4o-2024-05-13",
    }

    # Source: https://openai.com/pricing
    # Prices in $ per 1M tokens
    # Last updated: 2024-05-13
    PRICES = {
        "gpt-4-0613": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo-0613": {"input": 1.50, "output": 2.00},
        "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
        "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
        "gpt-4-1106-vision-preview": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-32k": {"input": 60.00, "output": 120.00},
        "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
        "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
        "gpt-3.5-turbo-16k-0613": {"input": 3.00, "output": 4.00},
        "whisper-1": {"input": 0.006, "output": 0.006},
        "tts-1": {"input": 15.00, "output": 15.00},
        "tts-hd-1": {"input": 30.00, "output": 30.00},
        "text-embedding-ada-002-v2": {"input": 0.10, "output": 0.10},
        "text-davinci-003": {"input": 20.00, "output": 20.00},
        "text-ada-001": {"input": 0.40, "output": 0.40},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
        "text-embedding-3-small": {"input": 0.02, "output": 0.02},
        "text-embedding-3-large": {"input": 0.13, "output": 0.13},
        "ada-v2": {"input": 0.10, "output": 0.10},
        "gpt-3.5-turbo-finetune": {"training": 8.00, "input": 3.00, "output": 6.00},
        "davinci-002-finetune": {"training": 6.00, "input": 12.00, "output": 12.00},
        "babbage-002-finetune": {"training": 0.40, "input": 1.60, "output": 1.60},
    }

    total_cost = 0.0  # class variable to persist total_cost

    def __init__(self) -> None:
        self.default_model = "gpt-3.5-turbo-0125"

    @classmethod
    def reset(cls) -> None:
        cls.total_cost = 0.0

    def get_total_cost(self) -> float:
        return CostEstimator.total_cost

    def __call__(self, function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            mock = kwargs.get('mock', True)
            model = kwargs.get('model', self.default_model)
            model = self.MODEL_SYNONYMS.get(model, model)
            completion_tokens = kwargs.get('completion_tokens', 1)

            messages = kwargs.get("messages")
            input_tokens = num_tokens_from_messages(messages, model=model)

            if mock:
                mock_output_message_content = lorem.words(completion_tokens) if completion_tokens else "Default response"
                mock_output_messages = {
                    'choices': [
                        {
                            'message': {
                                'content': mock_output_message_content
                            }
                        }
                    ]
                }
                output_tokens = num_tokens_from_messages([{"role": "assistant", "content": mock_output_message_content}], model=model)
                total_tokens = input_tokens + output_tokens
            else:
                response = function(*args, **kwargs)
                total_tokens = response.usage.total_tokens
                output_tokens = total_tokens - input_tokens

            input_cost = input_tokens * self.PRICES[model]['input'] / 1_000_000
            output_cost = output_tokens * self.PRICES[model]['output'] / 1_000_000
            cost = input_cost + output_cost
            CostEstimator.total_cost += cost  # update class variable

            # Display progress bar and cost with tqdm
            tqdm.write(f"Cost: ${cost:.4f} | Total: ${CostEstimator.total_cost:.4f}", end='\r')
            return mock_output_messages if mock else response
        return wrapper
