"""
Разреженное облако точек по набору изображений
"""
__version__ = '0.1'

# Оценка параметров камеры
FOCAL_EQ = 18.0
W, H = 24.0, 16.0  # Физические размеры сенсора в мм
INPUT_DIR = './desk/*.jpg'

import cv2 as cv
import numpy as np
import glob
import time
from collections import defaultdict
from Core.Export import to_ply, to_max


# cv.namedWindow('Debug', cv.WINDOW_NORMAL)


class Camera:
    def __init__(self):
        self.img = None
        self.kp = None  # Найденные особые точки
        self.des = None  # Дескрипторы
        self.P = None  # Матрица проекции (3x4)
        self.T = None  # Поворот + смещение (4x4)
        self.matches = defaultdict(dict)  # Хранит найденные соответствия в виде {точка: {камера: точка}}
        self.matches_3d = {}  # {точка на фото: точка в пространстве}


class Point3:
    def __init__(self):
        self.pt = None  # Координаты
        self.seen = 2  # Сколько камер её видят


cam_list = []
points_3d = []

if __name__ == "__main__":
    detector = cv.AKAZE_create()
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    path_list = glob.glob(INPUT_DIR)  # Получаем список изображений

    print('Initializing...')
    t0 = time.time()

    # Инициализируем камеры и ищем особые точки
    for path in path_list:
        cam = Camera()
        cam.img = cv.pyrDown(cv.imread(path, cv.IMREAD_COLOR))
        cam.kp, cam.des = detector.detectAndCompute(cam.img, None)
        cam_list.append(cam)

    t1 = time.time()
    print('Features found in {0:.3f}s'.format(t1 - t0))

    print('Matching features...')

    # Ищем соответствия
    for i in range(len(cam_list) - 1):
        for j in range(i + 1, len(cam_list)):
            matches = matcher.match(cam_list[i].des, cam_list[j].des)
            src, dst = [], []  # Координаты для фильтрации (см. ниже)
            i_kp, j_kp = [], []  # Индексы точек

            for m in matches:
                src.append(cam_list[i].kp[m.queryIdx].pt)  # Ищем точки из набора query в наборе train
                dst.append(cam_list[j].kp[m.trainIdx].pt)
                i_kp.append(m.queryIdx)
                j_kp.append(m.trainIdx)

            # Конвертация списка эффективнее прямого добавления значений к numpy-массиву
            src, dst = np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)

            # Дополнительно фильтруем ложные соответствия
            _, mask = cv.findFundamentalMat(src, dst, cv.FM_RANSAC, 3.0, 0.99)

            canvas = np.vstack((cam_list[i].img, cam_list[j].img))

            # Пишем соответствия в словарь
            for k in range(mask.shape[0]):
                if mask[k]:
                    cam_list[i].matches[i_kp[k]][j] = j_kp[k]
                    cam_list[j].matches[j_kp[k]][i] = i_kp[k]
                    
                    # Рисуем результат
                    cv.line(canvas, tuple(src[k].astype(int)), tuple(dst[k].astype(int) + np.array((0, cam_list[i].img.shape[0]), dtype=np.int32)), (255, 0, 0), 2)

            cv.imwrite("out/features{0}_{1}.png".format(i, j), canvas)
            # cv.imshow('Debug', canvas)
            # cv.waitKey(1)

    t2 = time.time()
    print('Feature matching took {0:.3f}s'.format(t2 - t1))

    # Восстанавливаем движение между парой соседних кадров
    cy, cx = (x / 2 for x in cam_list[0].img.shape[:-1])  # Центр изображения, он же principal point
    # Фокусное расстояние в пикселях (прямоугольный случай)
    fx = FOCAL_EQ * cam_list[0].img.shape[1] / W
    fy = FOCAL_EQ * cam_list[0].img.shape[0] / H

    # Матрица параметров камеры
    K = np.array(((fx, 0, cx),
                  (0, fy, cy),
                  (0, 0, 1)), dtype=np.float32)

    cam_list[0].T = np.identity(4, dtype=np.float32)
    cam_list[0].P = np.dot(K, np.array(((1, 0, 0, 0),
                                        (0, 1, 0, 0),
                                        (0, 0, 1, 0)), dtype=np.float32))

    for i in range(len(cam_list) - 1):
        prev, curr = cam_list[i], cam_list[i + 1]
        src, dst, used = [], [], []

        for k in range(len(prev.kp)):
            if i + 1 in prev.matches[k]:  # Если существует соответствие точке k на i+1-м кадре
                index = prev.matches[k][i + 1]
                src.append(prev.kp[k].pt)
                dst.append(curr.kp[index].pt)
                used.append(k)

        src, dst = np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)
        E, _ = cv.findEssentialMat(dst, src, K)
        ret, local_R, local_t, mask = cv.recoverPose(E, dst, src, K)

        T = np.hstack((local_R, local_t))
        T = np.vstack((T, np.array((0, 0, 0, 1), dtype=np.float32)))
        curr.T = np.dot(prev.T, T)  # Накапливаем преобразование

        R, t = curr.T[0:-1, 0:-1], curr.T[0:-1, 3]

        P = np.hstack((R.T, np.dot(-R.T, t).reshape((3, 1))))  # 0 = RC + t, C = -R.T * t, где C - центр камеры
        curr.P = np.dot(K, P)  # Новая матрица проекции

        points_4d = cv.triangulatePoints(prev.P, curr.P, src.T, dst.T).T

        # Восстановленный ветор смещения нормализован.
        # Масштабируем новые точки так, чтобы они совпали с уже найденными
        if i > 0:
            scale, count = 0, 0

            prev_cam = Point3()
            prev_cam.pt = prev.T[0:3, 3]

            new, existing = [], []

            for j in range(len(used)):
                k = used[j]
                if mask[j, 0] and (i + 1 in prev.matches[k]) and (k in prev.matches_3d):
                    pt = cv.convertPointsFromHomogeneous(points_4d[j].reshape((1, 4))).ravel()
                    index = prev.matches_3d[k]
                    average = points_3d[index].pt / (points_3d[index].seen - 1)

                    new.append(pt)
                    existing.append(average)

            # Отношения расстояний для ВСЕХ возможных пар точек
            # TODO: заменить на случайную выборку
            for j in range(len(new) - 1):
                for k in range(j + 1, len(new)):
                    s = cv.norm(existing[j] - existing[k], cv.NORM_L2) / cv.norm(new[j] - new[k], cv.NORM_L2)
                    scale += s
                    count += 1

            assert count > 0

            scale /= count
            print('frame:', i, 'scale:', scale, 'count:', count)

            local_t *= scale  # Масштабируем вектор и пересчитываем матрицы

            # TODO: разобраться с дублированием кода
            T = np.hstack((local_R, local_t))
            T = np.vstack((T, np.array((0, 0, 0, 1), dtype=np.float32)))
            curr.T = np.dot(prev.T, T)  # Накапливаем преобразование

            R, t = curr.T[0:-1, 0:-1], curr.T[0:-1, 3]

            P = np.hstack((R.T, np.dot(-R.T, t).reshape((3, 1))))  # 0 = RC + t, C = -R.T * t, где C - центр камеры
            curr.P = np.dot(K, P)  # Новая матрица проекции

            points_4d = cv.triangulatePoints(prev.P, curr.P, src.T, dst.T).T

        # Ищем хорошие точки
        for j in range(len(used)):
            if mask[j]:
                k = used[j]
                index = prev.matches[k][i + 1]
                pt = cv.convertPointsFromHomogeneous(points_4d[j].reshape((1, 4))).ravel()

                if k in prev.matches_3d:
                    curr.matches_3d[index] = prev.matches_3d[k]  # Нашли совпадение с уже найденной точкой
                    points_3d[prev.matches_3d[k]].pt += pt
                    points_3d[curr.matches_3d[index]].seen += 1
                else:
                    new_pt = Point3()  # Иначе добавляем новую точку
                    new_pt.pt = pt
                    points_3d.append(new_pt)

                    prev.matches_3d[k] = len(points_3d) - 1
                    curr.matches_3d[index] = len(points_3d) - 1

    # Усредняем положение точек
    for p in points_3d:
        if p.seen >= 3:
            p.pt /= p.seen - 1

    t3 = time.time()
    print('Reconstruction took {0:.3f}s'.format(t3 - t2))

    to_ply(points_3d, 'points.ply')
    to_max(cam_list, 'camera.ms', points=points_3d)
