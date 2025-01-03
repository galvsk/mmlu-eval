import subprocess
from dataclasses import dataclass
from typing import List


def get_api_key(api='claude'):
    if api == 'claude':
        api_str = 'claude-api-key'
    elif api == 'deepseek':
        api_str = 'deepseek-api-key'
    else:
        raise ValueError(f"API type : {api} not supported")

    cmd = ['security', 'find-generic-password', '-a', 'galvin.s.k@gmail.com', '-s', api_str, '-w']
    api_key = subprocess.check_output(cmd).decode('utf-8').strip()
    
    return api_key
