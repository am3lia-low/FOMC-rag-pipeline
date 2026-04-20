import re


def fix_encoding(text):
    if not isinstance(text, str):
        return ""

    try:
        fixed = text.encode('latin-1').decode('utf-8')
        return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    replacements = {
        '\u00e2\u0080\u0093': '\u2013',
        '\u00e2\u0080\u0094': '\u2014',
        '\u00e2\u0080\u0099': "'",
        '\u00e2\u0080\u009c': '\u201c',
        '\u00e2\u0080\u009d': '\u201d',
        '\u00e2\u0080\u0098': "'",
        '\u00c2\u00a0': ' ',
        '\u00e2\x80\x93': '\u2013',
        '\u00e2\x80\x94': '\u2014',
        '\u00e2\x80\x99': "'",
        '\u00e2\x80\x9c': '\u201c',
        '\u00e2\x80\x9d': '\u201d',
        '\u00e2': '\u2014',
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = fix_encoding(text)
    text = re.sub(r'(?<=[a-zA-Z])\d+(?=\s)', '', text)

    attendance_patterns = [
        r'\nAttendance\s*\n.*$',
        r'\nNotation Vote\s*\n.*$',
        r'\n_+\s*\n.*$',
    ]
    for pattern in attendance_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    text = re.sub(r'Implementation Note issued.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'For media inquiries.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()
