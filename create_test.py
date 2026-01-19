import json

result = []
with open("resources/docs.en.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    for key, value in data.items():
        if value["is_steady_state"] and (not value["is_script"]):
            comments = ""
            for c in value["comments"]:
                _c = c.removeprefix("{")
                if _c.endswith("}"):
                    _c = _c[:-1]
                comments += _c.strip()
            result.append((key, comments))

with open("test/items.txt", "w") as f:
    for key, comments in result:
        f.write(f"{key} --- {comments}\n")