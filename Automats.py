


def CDNF(x1, x2, x3, x4):
    return (not x1 and not x2 and not x3 and not x4) or (not x1 and x2 and x3 and x4) or (x1 and not x2 and x3) or (x1 and x2 and x3 and not x4)


# print(CDNF(0, 0, 0, 0))
# print(CDNF(0, 0, 0, 1))
# print(CDNF(0, 0, 1, 0))
# print(CDNF(0, 0, 1, 1))
# print(CDNF(0, 1, 0, 0))
# print(CDNF(0, 1, 0, 1))
# print(CDNF(0, 1, 1, 0))
# print(CDNF(0, 1, 1, 1))
# print(CDNF(1, 0, 0, 0))
# print(CDNF(1, 0, 0, 1))
# print(CDNF(1, 0, 1, 0))
# print(CDNF(1, 0, 1, 1))
# print(CDNF(1, 1, 0, 0))
# print(CDNF(1, 1, 0, 1))
# print(CDNF(1, 1, 1, 0))
# print(CDNF(1, 1, 1, 1))

def CKNF(x1, x2, x3, x4):
    return (x1 or x2 or x3 or not x4) and (x1 or x2 or not x3) and (x1 or not x2 or x3) and (x1 or not x2 or not x3 or x4) and (not x1 or x2 or x3) and (not x1 or not x2 or x3 or x4) and (not x1 or not x2 or not x4)

# print(CKNF(0, 0, 0, 0))
# print(CKNF(0, 0, 0, 1))
# print(CKNF(0, 0, 1, 0))
# print(CKNF(0, 0, 1, 1))
# print(CKNF(0, 1, 0, 0))
# print(CKNF(0, 1, 0, 1))
# print(CKNF(0, 1, 1, 0))
# print(CKNF(0, 1, 1, 1))
# print(CKNF(1, 0, 0, 0))
# print(CKNF(1, 0, 0, 1))
# print(CKNF(1, 0, 1, 0))
# print(CKNF(1, 0, 1, 1))
# print(CKNF(1, 1, 0, 0))
# print(CKNF(1, 1, 0, 1))
# print(CKNF(1, 1, 1, 0))
# print(CKNF(1, 1, 1, 1))


def ziga(x1, x2, x3, x4):
    res = 1 + x2 + x3 + x4 + x1*x3 + x2*x3 + x2*x4 + x3*x4 + x1*x2*x3 + x2*x3*x4 + x1*x2*x4 + x1*x2*x3*x4
    return res % 2

print(ziga(0, 0, 0, 0))
print(ziga(0, 0, 0, 1))
print(ziga(0, 0, 1, 0))
print(ziga(0, 0, 1, 1))
print(ziga(0, 1, 0, 0))
print(ziga(0, 1, 0, 1))
print(ziga(0, 1, 1, 0))
print(ziga(0, 1, 1, 1))
print(ziga(1, 0, 0, 0))
print(ziga(1, 0, 0, 1))
print(ziga(1, 0, 1, 0))
print(ziga(1, 0, 1, 1))
print(ziga(1, 1, 0, 0))
print(ziga(1, 1, 0, 1))
print(ziga(1, 1, 1, 0))
print(ziga(1, 1, 1, 1))