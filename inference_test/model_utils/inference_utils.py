from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from typing import List
import torch
import gc
from tqdm import tqdm
from accelerate import Accelerator

IMAGE_TOKEN_INDEX = -200

def batch_inference(
        model:LlavaForConditionalGeneration,
        processor:AutoProcessor,
        prompts:List[str],
        image_paths:List[str],
        batch_size:int,
        output_save_path:str,
        accelerator:Accelerator,
        **kwargs,
) -> list[str]:
    model.eval()
    generated_texts = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_images_paths = image_paths[i:i+batch_size]
            generated_text = process_batch(model,processor,batch_prompts, batch_images_paths,accelerator,**kwargs)
            generated_text = [gen_text.split("ASSISTANT:")[1].strip().replace("$}}%","") \
                       for gen_text in generated_text]
            
            with open(output_save_path, 'a') as f:
                f.write('\n'.join([gen_text.replace('\n','[[SEP]]') for gen_text in generated_text]) + '\n')

            generated_texts += generated_text
    return generated_texts

def process_batch(
        model:LlavaForConditionalGeneration,
        processor:AutoProcessor,
        batch_prompts:List[str],
        batch_images_paths:List[Image.Image],
        accelerator:Accelerator,
        **kwargs,
) -> list[str]:
    if len(batch_prompts) < 1: raise Exception("Batch size can't be under 1")
    if batch_images_paths[0] is not None: batch_images = [Image.open(image_path) for image_path in batch_images_paths]
    else: batch_images = None
    
    model_inputs = processor(batch_prompts, images=batch_images, padding=True, return_tensors="pt")
    if batch_images is not None: model_inputs = model_inputs.to(accelerator.device)
    try:
        # output = model.generate(**model_inputs,**kwargs)
        kwargs = {"max_new_tokens": 200, "do_sample": False}
        output = model.generate(**model_inputs,**kwargs)
        
        generated_texts = processor.batch_decode(output, skip_special_tokens=True)
    except KeyboardInterrupt as ke:
        print(ke)
        exit()
    except RuntimeError as re:
        print(re)
        if "CUDA" in str(re):
            del model_inputs
            if batch_images is not None: any(image.close() for image in batch_images)
            gc.collect()
            torch.cuda.empty_cache()

            temp_batch_size = len(batch_prompts)//2
            print("temp_batch_size:",temp_batch_size)

            temp_batch_prompts_1 = batch_prompts[:temp_batch_size]
            temp_batch_images_1 = batch_images[:temp_batch_size]
            generated_text_1 = process_batch(model,processor,temp_batch_prompts_1,temp_batch_images_1,accelerator,**kwargs)

            temp_batch_prompts_2 = batch_prompts[temp_batch_size:]
            temp_batch_images_2 = batch_images[temp_batch_size:]
            generated_text_2 = process_batch(model,processor,temp_batch_prompts_2,temp_batch_images_2,accelerator,**kwargs)

            generated_texts = generated_text_1 + generated_text_2
    else:
        del model_inputs
        gc.collect()
        torch.cuda.empty_cache()
        if batch_images is not None: any(image.close() for image in batch_images)
    return generated_texts