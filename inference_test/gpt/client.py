from openai import OpenAI

def get_client(api_key:any = None):
    if api_key is None: api_key = 'sk-proj-3ARwM8d3jhDa5Ad9WHd1T3BlbkFJtYEDdkCLsopZaeudHCBP'
    return OpenAI(api_key=api_key)