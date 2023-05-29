
"""
Экспорт и всё, что с ним связано
"""
__version__ = '0.1'

# Для экспорта облака точек в PLY
PLY_HEADER = '''ply
format ascii 1.0
element vertex {}
property float32 x
property float32 y
property float32 z
end_header
'''

MAX_HEADER = '''point cross:off centermarker:on pos:[0.0, 0.0, 0.0] name:"origin"
animate on (
cam = freecamera name:"Camera"
cam.nearclip = 0.0
cam.parent = $origin
'''

MAX_TEMPLATE = '''at time {0}f cam.rotation = quat {1}
at time {0}f cam.position = {2}\n'''

import numpy as np


def to_ply(points, out):
    '''
    Экспорт облака точек в текстовый файл
    :param points: список точек (Core.Matchmover.Point3)
    :param out: имя файла
    :return:
    '''
    with open(out, 'w') as f:
        f.write(PLY_HEADER.format(len(points)))
        for p in points:
            f.write(np.array2string(p.pt).strip(' []') + '\n')


def mat2quat(R):
    '''
    Преобразование матрицы поворота в кватернион
    :param R: матрица поворота 3х3
    :return: numpy-массив 1х4 - кватернион
    '''
    quat = np.zeros(4, dtype=np.float32)
    nxt = (1, 2, 0)
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        s = (tr + 1) ** 0.5
        quat[0] = s / 2
        s = 0.5 / s
        quat[1] = (R[1, 2] - R[2, 1]) * s
        quat[2] = (R[2, 0] - R[0, 2]) * s
        quat[3] = (R[0, 1] - R[1, 0]) * s
    else:
        # Если сумма диагоналей равна нулю, то вычисления зависят от того, в каком столбце расположен
        # максимальный элемент диагонали
        i = 0
        if R[1, 1] > R[0, 0]: i = 1
        if R[2, 2] > R[i, i]: i = 2
        j = nxt[i]
        k = nxt[j]

        s = (R[i, i] - (R[j, j] + R[k, k]) + 1) ** 0.5
        quat[i] = s * 0.5
        quat[3] = (R[j, k] - R[k, j]) * s
        quat[j] = (R[i, j] + R[j, i]) * s
        quat[k] = (R[i, k] + R[k, i]) * s

    return quat


def to_max(cameras, out, points=None, scale=1):
    with open(out, 'w') as f:
        f.write(MAX_HEADER)
        
        for i, c in enumerate(cameras):
            quat = np.array2string(mat2quat(c.T[0:3, 0:3]), precision=5, sign='-').strip(' []')
            tmp = c.T
            print(tmp)
            pos = np.array2string(c.T[0:3, 3].T, precision=5, separator=',', sign='-')
            f.write(MAX_TEMPLATE.format(i, quat, pos))

        f.write(')\n')

        if points is not None:
            for i, p in enumerate(points):
                pos = np.array2string(p.pt, precision=5, separator=',')
                f.write('point cross:off centermarker:on pos:{} name:"point_{}" parent:$origin\n'.format(pos, i))
