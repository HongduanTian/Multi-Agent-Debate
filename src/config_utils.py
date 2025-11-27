import yaml

class LLMConfig:
    def __init__(self, general_config: dict, model_config: dict):
        self.model = model_config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        self.model_path = model_config.get("model_path", None)
        self.max_token_length = model_config.get("max_token_length", general_config.get("max_token_length", 24064))
        self.temperature = model_config.get("temperature", general_config.get("temperature", 1))
        self.top_p = general_config.get("top_p", 1)


def load_configs_from_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)