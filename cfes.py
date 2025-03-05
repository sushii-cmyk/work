import random

D = {
    "A": ["a", "the"],
    "N": ["A dog"],
    "V": ["sleeps"]
}


s = "N V"

while not D.keys().isdisjoint(s):
    print(s)
    for k, v in D.items():
        if k in s:
            s = s.replace(k, random.choice(v))

print(s)