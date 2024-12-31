import subprocess
from dataclasses import dataclass
from typing import List


def get_api_key():
    cmd = ['security', 'find-generic-password', '-a', 'galvin.s.k@gmail.com', '-s', 'claude-api-key', '-w']
    api_key = subprocess.check_output(cmd).decode('utf-8').strip()
    
    return api_key
