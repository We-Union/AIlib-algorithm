def txtread(txt, encoding="utf-8", split_by_line=False):
    with open(txt, "r", encoding=encoding) as f:
        text = f.read()
    if split_by_line:
        text = text.split("\n")
    return text