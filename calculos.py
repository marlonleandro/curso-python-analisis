def area_base_altura(base, altura):
    return (base * altura) / 2

def area_tres_lados(a, b, c):
    s = (a + b + c) / 2
    return (s * (s - a) * (s - b) * (s - c)) ** 0.5
