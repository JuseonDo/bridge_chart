from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model(model_name:str, quantization_config):
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

    return model, processor