
class TokenUsageTracker:
    """Tracks token usage and calculates costs"""
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        self.usage_history = []
    
    def update_usage(self, model:str, input_tokens: int, output_tokens: int):
        
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        self.usage_history.append(usage_record)
    
    def get_summary(self):
        summary = {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }
        return summary