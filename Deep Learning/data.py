import random

for i in range(240):
    a, b, c, d = random.randint(1, 15), random.randint(2, 5), random.randint(3, 10), random.randint(0, 1)
    if (10 <= a <= 20) and (b == 2) and (3 <= c <= 8):
        y = 2
    elif (10 <= a <= 20) and (b == 3) and (3 <= c <= 8):
        y = 2
    elif (10 <= a <= 20) and (b == 4) and (4 <= c <= 8) and (d == 0):
        y = 1
    elif (10 <= a <= 20) and (b == 4) and (4 <= c <= 8) and (d == 1):
        y = 0
    elif (10 <= a <= 20) and (b == 5) and (4 <= c <= 10):
        y = 0
    elif (1 <= a <= 9) and (b == 2) and (3 <= c <= 6):
        y = 2
    elif (1 <= a <= 9) and (b == 3) and (3 <= c <= 6):
        y = 2
    elif (3 <= a <= 9) and (b == 4) and (3 <= c <= 10):
        y = 1
    elif (3 <= a <= 9) and (b == 5) and (5 <= c <= 10):
        y = 0
    elif (1 <= a <= 3) and (b == 2):
        y = 2
    else:
        y = 1
    s = f"(np.array([[{a}., {b}., {c}., {d}.]]), {y}),"
    print(s)