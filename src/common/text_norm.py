def normalize_fa(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    repl = {"\u200c":" ", "ي":"ی", "ك":"ک", "‌":" ", "ٔ":"", "ـ":""}
    for k, v in repl.items(): s = s.replace(k, v)
    return " ".join(s.split())
