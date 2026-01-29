import random
import re

def process_wildcards(text: str) -> str:
    # regex to find patterns like {option1|option2|option3}
    pattern = r"\{([^{}]+)\}"
    
    def replace_match(match):
        # get the content (e.g red|green|blue)
        options = match.group(1).split('|')
        return random.choice(options).strip()
    
    while re.search(pattern, text):
        text = re.sub(pattern, replace_match, text)
    
    return text