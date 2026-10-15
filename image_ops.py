"""Михаил: Часть 2: преобразования, размытие и карта контрастности.

Только собственные функции и обычные циклы.
NumPy/OpenCV не используются.
Все операции возвращают новый массив.
Входной массив остаётся без изменений.
"""

from statistics_ops import RGBImage, ScalarImage, clamp, intensity_matrix

NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS_8 = NEIGHBORS_4 + ((-1, -1), (1, -1), (-1, 1), (1, 1))


def point_transform(
    image: RGBImage, operation: str, parameter: float = 0,
    channel: int = -1,
) -> RGBImage:
    """Точечные операции; channel=-1 означает все три RGB-канала.

    brightness: c' = c + parameter.
    contrast: c' = mean + parameter * (c - mean).
    negative: c' = 255 - c.
    gamma: c' = 255 * (c / 255) ** parameter.
    """
    if channel not in (-1, 0, 1, 2):
        raise ValueError('Неизвестный канал.')
    if operation not in ('brightness', 'contrast', 'negative', 'gamma'):
        raise ValueError('Неизвестная операция.')
    if operation == 'contrast' and parameter < 0:
        raise ValueError(
            'Коэффициент контраста должен быть неотрицательным.',
        )
    if operation == 'gamma' and parameter <= 0:
        raise ValueError('Гамма должна быть положительной.')

    center = 0.0
    if operation == 'contrast':
        total = 0.0
        for row in image:
            for pixel in row:
                if channel == -1:
                    total += (pixel[0] + pixel[1] + pixel[2]) / 3.0
                else:
                    total += pixel[channel]
        center = total / (len(image) * len(image[0]))

    result = []
    for row in image:
        result_row = []
        for pixel in row:
            output = []
            for index in range(3):
                value = pixel[index]
                if channel == -1 or channel == index:
                    if operation == 'brightness':
                        value += parameter
                    elif operation == 'contrast':
                        value = center + parameter * (value - center)
                    elif operation == 'negative':
                        value = 255 - value
                    else:
                        value = 255 * (value / 255.0) ** parameter
                output.append(clamp(value))
            result_row.append((output[0], output[1], output[2]))
        result.append(result_row)
    return result


# Урок 1: напиши здесь swap_channels(image, first, second)
# и mirror(image, horizontal). См. learning/lessons/0001-image-as-data.html


def blur(image: RGBImage, connectivity: int) -> RGBImage:
    """Среднее центра и 4/8 соседей.

    Результат записывается отдельно.
    """
    if connectivity not in (4, 8):
        raise ValueError('Связность должна быть 4 или 8.')
    neighbors = NEIGHBORS_4 if connectivity == 4 else NEIGHBORS_8
    offsets = ((0, 0),) + neighbors
    height, width = len(image), len(image[0])
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            totals = [0, 0, 0]
            count = 0
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    count += 1
                    for channel in range(3):
                        totals[channel] += image[ny][nx][channel]
            row.append((clamp(totals[0] / count),
                        clamp(totals[1] / count),
                        clamp(totals[2] / count)))
        result.append(row)
    return result


def integral_image(values: ScalarImage) -> ScalarImage:
    """Таблица сумм: table[y][x] содержит прямоугольник выше и левее."""
    height, width = len(values), len(values[0])
    table = [[0.0] * (width + 1) for _ in range(height + 1)]
    for y in range(height):
        row_sum = 0.0
        for x in range(width):
            row_sum += values[y][x]
            table[y + 1][x + 1] = table[y][x + 1] + row_sum
    return table


def contrast_map(
    image: RGBImage, neighborhood: str, size: int = 11
) -> tuple[ScalarImage, float]:
    """C(p)=abs(I(p)-mean(neighbors)), C(I)=mean(C(p)), лекция 2_1.

    Центр исключён из всех окрестностей.
    У границ берём реальных соседей.
    Для окна таблица сумм ускоряет тот же ручной расчёт среднего.
    """
    if neighborhood not in ('4', '8', 'window'):
        raise ValueError('Неизвестная окрестность.')
    if neighborhood == 'window' and (size < 3 or size % 2 == 0):
        raise ValueError('Размер окна должен быть нечётным и не меньше 3.')
    values = intensity_matrix(image)
    height, width = len(image), len(image[0])
    table = integral_image(values) if neighborhood == 'window' else []
    offsets = NEIGHBORS_4 if neighborhood == '4' else NEIGHBORS_8
    result = []
    contrast_total = 0.0
    radius = size // 2
    for y in range(height):
        row = []
        for x in range(width):
            neighbor_sum = 0.0
            count = 0
            if neighborhood == 'window':
                left, right = max(0, x - radius), min(width, x + radius + 1)
                top, bottom = max(0, y - radius), min(height, y + radius + 1)
                neighbor_sum = (table[bottom][right] - table[top][right]
                                - table[bottom][left] + table[top][left]
                                - values[y][x])
                count = (right - left) * (bottom - top) - 1
            else:
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor_sum += values[ny][nx]
                        count += 1
            contrast = (abs(values[y][x] - neighbor_sum / count)
                        if count else 0.0)
            row.append(contrast)
            contrast_total += contrast
        result.append(row)
    return result, contrast_total / (height * width)
