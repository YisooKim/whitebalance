# -*- coding: utf-8 -*-
#강무성, 오태권, 한희일 (2006). 새로운 Automatic White Balance 알고리즘의 제안. 대한전자공학회 학술대회, 719-722
# 위 논문 참고

import cv2
import numpy as np

#히스토그램 평활화
def equ_Hist(img):
    b, g, r = cv2.split(img)
    e_r = cv2.equalizeHist(r);
    e_g = cv2.equalizeHist(g);
    e_b = cv2.equalizeHist(b);
    equ = cv2.merge([e_b, e_g, e_r])
    return equ

# 수식에 따라 흰 색 영역이라고 생각하는 부분의 좌표 추출.
def extractWhite(img):
    b, g, r = cv2.split(img)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.731 * (r - y)

    mask1 = (y <= 200)
    mask2 = (cb <= -2) | (cb >= 2)
    mask3 = (cr <= -2) | (cr >= 2)

    mask = (mask1 | mask2 | mask3)
    mask = np.logical_not(mask)

    idx= np.transpose(np.nonzero(mask))
    return idx

#화이트밸런스 기능
def WB(idx,img):
    b, g, r = cv2.split(img)
    equ= equ_Hist(img) #평활화
    b1, g1, r1 = cv2.split(equ)
    y = 0.299 * r1 + 0.587 * g1 + 0.114 * b1

    # 가장 밝은 영역의 y 좌표 추출.
    result = np.where(y == np.amax(y))

    # 밝은 영역의 평균값.
    listOfCordinates = list(zip(result[0], result[1]))
    for cord in listOfCordinates:
        i,j = cord[0], cord[1]
        y_avg = (int(r[i][j])+int(g[i][j])+int(b[i][j]))/3

    count=idx.shape[0]
    r_sum=0
    g_sum=0
    b_sum=0

    for k in idx:
        i,j=k[0],k[1]
        r_sum += r[i][j]
        g_sum += g[i][j]
        b_sum += b[i][j]

    r_avg = r_sum / count
    g_avg = g_sum / count
    b_avg = b_sum / count

    r_gain = y_avg/r_avg
    g_gain = y_avg/g_avg
    b_gain = y_avg/b_avg



    while r_gain + b_gain + g_gain > 3:
        gain_sum = r_gain + b_gain + g_gain
        r_gain = (3 * r_gain)/gain_sum
        g_gain = (3 * g_gain)/gain_sum
        b_gain = (3 * b_gain)/gain_sum

    # 기존의 값에 gain값 곱함
    r=r*r_gain
    b=b*b_gain
    g=g*g_gain

    #overflow 방지
    r=r.astype(np.uint8)
    g=g.astype(np.uint8)
    b=b.astype(np.uint8)

    return cv2.merge([b,g,r])
if __name__ == "__main__":
    file = 'iu.jpg'
    img = cv2.imread(file)

    equ= equ_Hist(img)
    idx= extractWhite(equ)
    equ2= WB(idx,img)

    cv2.imshow('a', img)
    cv2.imshow('result2', equ2)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

