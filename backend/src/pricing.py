PRICING = {
    "llama-3.3-70b-versatile": {
        "input_per_mtok":  0.59,
        "output_per_mtok": 0.79,
    },
    "llama3-8b-8192": {
        "input_per_mtok":  0.05,
        "output_per_mtok": 0.08,
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    p = PRICING.get(model, {"input_per_mtok": 0.0, "output_per_mtok": 0.0})
    return {
        "input_cost_usd":  (input_tokens  / 1_000_000) * p["input_per_mtok"],
        "output_cost_usd": (output_tokens / 1_000_000) * p["output_per_mtok"],
    }
