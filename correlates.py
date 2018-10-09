# Корреляция по Пирсону
def correlate(samp1, samp2):
    if len(samp1) != len(samp2):
        return "error data different range"
    size = len(samp1)
    avg1 = samp1.mean()
    avg2 = samp2.mean()
    sums1, sums2, sums3 = 0, 0, 0
    for i in range(size):
        sums1 += (samp1[i] - avg1) * (samp2[i] - avg2)
    for i in range(size):
        sums2 += (samp1[i] - avg1) ** 2
    for i in range(size):
        sums3 += (samp2[i] - avg1) ** 2
    return sums1 / math.sqrt(sums2 * sums3)


def sort_un(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1, i, -1):
            if arr[j][1] < arr[j - 1][1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]

    return arr


# Корреляция по Спирмену
def correlate2(samp1, samp2):
    united = []  # Список для объединённой выборки
    for i in range(len(samp1)):
        united.append(("x", samp1[i], i))
    for i in range(len(samp2)):
        united.append(("y", samp2[i], i))
    rang_x = [i for i in range(len(samp1))]
    rang_y = [i for i in range(len(samp2))]
    united = sort_un(united)

    for i in range(len(united)):
        if united[i][0] == "x":
            rang_x[united[i][2]] = i + 1
        if united[i][0] == "y":
            rang_y[united[i][2]] = i + 1

    rang_x = np.array(rang_x)
    rang_y = np.array(rang_y)
    return correlate(rang_x, rang_y)
