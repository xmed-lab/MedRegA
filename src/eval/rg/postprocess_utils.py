import re
import string

contractions = {
    "pa": 'posterior anterior',
    'ivc': 'inferior vena cava',
    'head': 'brain',
    'both': 'bilateral',
    'tumour': 'tumor',
    'tumours': 'tumors',
}

def convert_contract(x):
    if x in contractions:
        return contractions[x]
    else:
        return x

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    # ",",
    "?",
    "!",
]


def cleanStr(token):
    parts = re.split(r'[,:.\n]', token.lower())
    filtered_parts = []
    for part in parts:
        part = part.lstrip().rstrip()
        if part != '' and '_' not in part:
            filtered_parts.append(part)
        
    token = '.'.join(filtered_parts)
    token = re.sub(r'\s+', ' ', token).lstrip().rstrip()
    
    return token

def cleanStr_zh(token):
    token = re.sub(r'\[.*?\]', '', token).replace('[', ' ').replace(']', '')
    token = token.replace('-', '')
    parts = re.split(r'[\n]', token.lower())
    filtered_parts = []
    for part in parts:
        part = re.sub(r'\d+\.', '', part)
        part = part.lstrip().rstrip()
        if part != '':
            filtered_parts.append(part)
    token = '，'.join(filtered_parts)
    token = token.replace('：，', '：').replace('。，', '，')
    return token

def remove_punctuation(input_string: str):
    # Make a translator object to replace punctuation with none
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    # Use the translator
    return input_string.translate(translator)

def unify_comma(input_string: str):
    # Replace commas followed by any non-letter characters up to the next letter with a single comma.
    pattern = r',[^a-zA-Z]*'
    result = re.sub(pattern, ',', input_string)
    return result
