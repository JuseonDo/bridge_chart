import os

def get_drafts(draft_path:str):
    if os.path.exists(draft_path):
        print('-'*40)
        print('*** Using Draft Mode ***')
        print('draft_path:',draft_path)
        print('-'*40)
        with open(draft_path) as f:
            return [line.replace('[[SEP]]','\n').strip() for line in f]
    else:
        print('Not Using Draft')
        return None