#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
import cv2

# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask

def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    print(region_size)

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True

    
    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            #print(x,y)
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()
    
    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target

def test():
    '''
    img_mask = np.asarray(PIL.Image.open('./mask/test1_mask.png').copy())#mask_heart_left.jpg'))
    img_mask.flags.writeable = True
    img_source = np.asarray(PIL.Image.open('./source/test1_src.png').copy())#source_heart_left.jpg'))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open('./target/test1_target.png').copy())#target_heart_left.jpg'))
    img_target.flags.writeable = True
    '''

    for i in range(1, 13):
        if i==11:
            continue
        # target은 고정
        img_target=cv2.imread("img6.jpg", cv2.IMREAD_COLOR)        
        #img_target=cv2.add(img_target, (50, 50, 50, 0)) # 밝기 조절

        # 경로 설정
        m1="final_eyebrows/"+str(i)+"_l_m.jpg"
        s1="final_eyebrows/"+str(i)+"_l.png"
        m2="final_eyebrows/"+str(i)+"_r_m.jpg"
        s2="final_eyebrows/"+str(i)+"_r.png"

        # 읽어들이기
        img_mask1=cv2.imread(m1, cv2.IMREAD_COLOR)
        img_source1=cv2.imread(s1, cv2.IMREAD_COLOR)
        img_mask2=cv2.imread(m2, cv2.IMREAD_COLOR)
        img_source2=cv2.imread(s2, cv2.IMREAD_COLOR)

        # target 눈썹 영역 bounding box 크기만큼 resize
        img_mask1=cv2.resize(img_mask1, (73, 26)) # 왼쪽 # 70 24
        img_source1=cv2.resize(img_source1, (73,26))
        img_mask2=cv2.resize(img_mask2, (76, 26)) # 오른쪽 # 71 24
        img_source2=cv2.resize(img_source2, (76,26))

        # 색 영역 BGR -> RGB
        img_mask1 = cv2.cvtColor(img_mask1, cv2.COLOR_BGR2RGB)
        img_source1 = cv2.cvtColor(img_source1, cv2.COLOR_BGR2RGB)
        img_mask2 = cv2.cvtColor(img_mask2, cv2.COLOR_BGR2RGB)
        img_source2 = cv2.cvtColor(img_source2, cv2.COLOR_BGR2RGB)
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)

        # target 눈썹 시작 위치를 offset으로 설정 (행, 열)
        img_ret = blend(img_target, img_source1, img_mask1, offset=(162,145)) # y, x 171 126
        img_ret = blend(img_ret, img_source2, img_mask2, offset=(161,249)) # y, x 172 228

        img_ret = PIL.Image.fromarray(np.uint8(img_ret))
        svpth="result_more/more_img6_result_"+str(i)+".jpg"
        img_ret.save(svpth)

if __name__ == '__main__':
    test()