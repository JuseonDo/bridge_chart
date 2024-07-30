from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt_request(client:OpenAI, image_path:any, instruction:str, model_name:str = 'gpt-4o') -> str:
    if image_path is not None:
        base64_image = encode_image(image_path)
        messages = [
                    {
                        "role": "user", 
                        "content": [
                                    {"type": "text", "text": instruction},
                                    {"type": "image_url", "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"}
                                    }
                                ]
                    }
                ]
    else:
        messages = [
                    {
                        "role": "user", 
                        "content": [
                                    {"type": "text", "text": instruction}
                                ]
                    }
                ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content

def output_save(save_path, output):
    output = output.replace('\n','[[SEP]]')
    with open(save_path, 'a') as f:
        f.write(output + '\n')


def geval():
    return