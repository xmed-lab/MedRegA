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
    parts = re.split(r'[,:.]', token)
    filtered_parts = [part for part in parts if '_' not in part]
    token = ','.join(filtered_parts).lower()
    
    parts = re.split(r'[,:.]', token)
    filtered_parts = [' '.join([convert_contract(p) for p in part.split(' ')]) for part in parts if '_' not in part]
    token = ','.join(filtered_parts).replace('-', ' ')
    
    # if '1' in token:
    #     token = 'one'
    
    # token = token.lower()
    # _token = token
    # for p in punct:
    #     if (p + " " in token or " " + p in token) or (
    #         re.search(comma_strip, token) != None
    #     ):
    #         _token = _token.replace(p, "")
    #     else:
    #         _token = _token.replace(p, " ")
    # token = period_strip.sub("", _token, re.UNICODE)

    # _token = []
    # temp = token.lower().split()
    # for word in temp:
    #     word = manual_map.setdefault(word, word)
    #     if word not in articles:
    #         _token.append(word)
    # for i, word in enumerate(_token):
    #     if word in contractions:
    #         _token[i] = contractions[word]
    # token = " ".join(_token)
    # token = token.replace(",", "")
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
