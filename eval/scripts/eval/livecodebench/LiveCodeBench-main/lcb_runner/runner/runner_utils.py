from lcb_runner.lm_styles import LMStyle, LanguageModel


def build_runner(args, model: LanguageModel):
    if model.model_style == LMStyle.OpenAIChat:
        from lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style == LMStyle.OpenAIReason:
        from lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style == LMStyle.Gemini:
        from lcb_runner.runner.gemini_runner import GeminiRunner

        return GeminiRunner(args, model)
    if model.model_style == LMStyle.Claude3:
        from lcb_runner.runner.claude3_runner import Claude3Runner

        return Claude3Runner(args, model)
    if model.model_style == LMStyle.Claude:
        from lcb_runner.runner.claude_runner import ClaudeRunner

        return ClaudeRunner(args, model)
    if model.model_style == LMStyle.MistralWeb:
        from lcb_runner.runner.mistral_runner import MistralRunner

        return MistralRunner(args, model)
    if model.model_style == LMStyle.CohereCommand:
        from lcb_runner.runner.cohere_runner import CohereRunner

        return CohereRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    elif model.model_style in []:
        raise NotImplementedError(
            f"Runner for language model style {model.model_style} not implemented yet"
        )
    else:
        from lcb_runner.runner.vllm_runner import VLLMRunner

        return VLLMRunner(args, model)
