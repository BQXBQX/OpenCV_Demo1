import cv2
import cv2 as cv
import numpy as np

def cv_show(img, imgName='Image'):
    cv.imshow(imgName, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread("assets/Demo1.jpg")

# 获取图像的宽度和高度
height, width = img.shape[:2]

# 缩小图像尺寸
new_width = int(width / 2)
new_height = int(height / 2)
resized_image = cv.resize(img, (new_width, new_height))





# cv_show("image",resized_image)

# 调色板操作

# def nothing(x):
#     pass
# # 创建一个黑色的图像，一个窗口
# img = np.zeros((300,512,3), np.uint8)
# cv.namedWindow('image')
# # 创建颜色变化的轨迹栏
# cv.createTrackbar('R','image',0,255,nothing)
# cv.createTrackbar('G','image',0,255,nothing)
# cv.createTrackbar('B','image',0,255,nothing)
# # 为 ON/OFF 功能创建开关
# switch = '0 : OFF \n1 : ON'
# cv.createTrackbar(switch, 'image',0,1,nothing)
# while(1):
#     cv.imshow('image',img)
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     # 得到四条轨迹的当前位置
#     r = cv.getTrackbarPos('R','image')
#     g = cv.getTrackbarPos('G','image')
#     b = cv.getTrackbarPos('B','image')
#     s = cv.getTrackbarPos(switch,'image')
#     if s == 0:
#         img[:] = 0
#     else:
#         img[:] = [b,g,r]
# cv.destroyAllWindows()




#为图像设置边框（填充）
# BLUE = [255,0,0]
# # img1 = cv.imread('assets/Demo1.jpg')
# replicate = cv.copyMakeBorder(resized_image,10,10,10,10,cv.BORDER_REPLICATE)
# reflect = cv.copyMakeBorder(resized_image,10,10,10,10,cv.BORDER_REFLECT)
# reflect101 = cv.copyMakeBorder(resized_image,10,10,10,10,cv.BORDER_REFLECT_101)
# wrap = cv.copyMakeBorder(resized_image,10,10,10,10,cv.BORDER_WRAP)
# constant= cv.copyMakeBorder(resized_image,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
#
# #
# # print(constant.shape)
# # print(resized_image.shape)
#
# def compareBorder(image):
#
#     new_resized_image = cv.resize(image,(682,1024))
#
#     combined_image = np.concatenate((new_resized_image,resized_image),axis=1)
#     cv_show("combined_image",combined_image)
#
# compareBorder(reflect)




# #图像的算数运算
# #注意 OpenCV加法和Numpy加法之间有区别。OpenCV加法是饱和运算，而Numpy加法是模运算。
# x = np.uint8([250])
# y = np.uint8([10])
# print( cv.add(x,y) ) # 250+10 = 260 => 255
# [[255]]
# print( x+y )          # 250+10 = 260 % 256 = 4
# [4]
#
# #图片的融合
# img2 = cv.imread('assets/Demo2.jpg')
# print(resized_image.shape)
# resized_image2 =  cv.resize(img2,(682,1024))
# # dst = cv.addWeighted(resized_image,0.7,resized_image2,0.3,0)
#
# # cv_show('dst',dst)
#
# #换位运算
# # 加载两张图片
# # 我想把logo放在左上角，所以我创建了ROI
# # cv_show('img',img2)
# resized_image3 = cv.resize(img2,(150,225))
# # print(resized_image3.shape)
# # cv_show('image',resized_image3)
# rows,cols,channels = resized_image3.shape
# roi = resized_image[0:rows, 0:cols ]
# # cv_show(roi)
# # cv_show(resized_image)
# # # 现在创建logo的掩码，并同时创建其相反掩码
# img2gray = cv.cvtColor(resized_image3,cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 150, 255, cv.THRESH_BINARY)
# mask_inv = cv.bitwise_not(mask)
# # cv_show(mask_inv)
# # # 现在将ROI中logo的区域涂黑
# img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# # # 仅从logo图像中提取logo区域
# img2_fg = cv.bitwise_and(resized_image3,resized_image3,mask = mask)
# # # 将logo放入ROI并修改主图像
# dst = cv.add(img1_bg,img2_fg)
# resized_image[0:rows, 0:cols ] = dst
# cv_show(resized_image)

#竖直偏移
#您可以将其放入**np.float32**类型的Numpy数组中，并将其传递给**cv.warpAffine**函数。
# 参见下面偏移为(100, 50)的示例：

rows,cols = resized_image.shape[:2]
# M = np.float32([[1,0,100],[0,1,50]])
# dst = cv.warpAffine(resized_image,M,(cols,rows))
#
# cv_show(dst)

#旋转
# M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
# dst = cv.warpAffine(resized_image,M,(cols,rows))
#
# cv_show(dst)

#仿射变换
# rows,cols,ch = resized_image.shape
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])
# M = cv.getAffineTransform(pts1,pts2)
# dst = cv.warpAffine(resized_image,M,(cols,rows))
#
# cv_show(dst)

#透视变换
# rows,cols,ch = resized_image.shape
# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# M = cv.getPerspectiveTransform(pts1,pts2)
# dst = cv.warpPerspective(resized_image,M,(rows,cols))
#
# cv_show(dst)

#图像的梯度


# laplacian = cv.Laplacian(resized_image,cv.CV_64F)
# sobelx = cv.Sobel(resized_image,cv.CV_64F,1,0,ksize=5)
# sobely = cv.Sobel(resized_image,cv.CV_64F,0,1,ksize=5)
#
# # 取Laplacian、Sobel x和Sobel y的绝对值
# laplacian_abs = np.abs(laplacian)
# sobelx_abs = np.abs(sobelx)
# sobely_abs = np.abs(sobely)
#
# # 将绝对值转换回cv.CV_8U类型
# laplacian_abs_uint8 = cv.convertScaleAbs(laplacian_abs)
# sobelx_abs_uint8 = cv.convertScaleAbs(sobelx_abs)
# sobely_abs_uint8 = cv.convertScaleAbs(sobely_abs)
#
# # 将Sobel x和Sobel y结合起来
# sobel_combined = cv.addWeighted(sobelx_abs_uint8, 0.5, sobely_abs_uint8, 0.5, 0)
#
# cv_show(laplacian_abs_uint8)
# cv_show(sobel_combined)
#
# # 使用Scharr算子计算梯度
# scharrx = cv.Scharr(resized_image, cv.CV_64F, 1, 0)
# scharry = cv.Scharr(resized_image, cv.CV_64F, 0, 1)
#
# # 取Scharr x和Scharr y的绝对值
# scharrx_abs = np.abs(scharrx)
# scharry_abs = np.abs(scharry)
#
# # 将绝对值转换回cv.CV_8U类型
# scharrx_abs_uint8 = cv.convertScaleAbs(scharrx_abs)
# scharry_abs_uint8 = cv.convertScaleAbs(scharry_abs)
#
# # 将Scharr x和Scharr y结合起来
# scharr_combined = cv.addWeighted(scharrx_abs_uint8, 0.5, scharry_abs_uint8, 0.5, 0)
#
# cv_show(scharr_combined)



# #Canny边缘检测
# gray = cv.cvtColor(resized_image,cv.COLOR_BGR2GRAY)
# # cv_show(gray)
# # print(gray.shape)
# edges = cv.Canny(gray,50,250)
# edges1 = cv2.Canny(gray,125,175)
#
# combined_image = np.concatenate((edges,edges1),axis=1)

# cv_show(combined_image)


# #轮廓
# # ret, thresh = cv.threshold(gray, 127, 255, 0)
# # cv_show(thresh)
# #这里使用CHAIN_APPROX_SIMPLE可以有效的减少内存
# contours, hierarchy = cv.findContours(edges1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# image_copy = resized_image.copy()
# cv.drawContours(image_copy, contours, -1, (0,255,0), 1)
#
# # print(len(contours[80]))
# combined_image = np.concatenate((image_copy,resized_image),axis=1)
#
# # cv_show(thresh)
# # cv_show(combined_image)



# #模板匹配
# template = cv.imread('assets/roi.jpg',0)
#
# gray = cv.cvtColor(resized_image,cv.COLOR_BGR2GRAY)
#
# gray2 = gray.copy()
#
# w, h = template.shape[::-1]
#
#
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
#
# for meth in methods:
#     gray = gray2.copy()
#     #使用eval()函数将方法名称字符串转换为对应的OpenCV常量，
#     # 并将结果存储在method变量中，以便在后续的模板匹配步骤中使用。
#     method = eval(meth)
# #     # 应用模板匹配
#     res = cv.matchTemplate(gray,template,method)
#     print(res)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# #     # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     # 这行代码使用top_left和bottom_right坐标定义一个矩形区域，然后使用cv.rectangle()
#     # 函数在图像img上绘制该矩形框。矩形框的颜色为255（白色），线宽为2。
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(gray,top_left, bottom_right, (0, 255, 0), 2)
#
#     cv_show(gray)
#
# # cv_show(gray)


#
#实战example
from imutils import contours
import imutils


img_ref  = cv.imread('assets/model.png')

img_ref_gray = cv2.cvtColor(img_ref , cv2.COLOR_BGR2GRAY)
_,img_ref_gray = cv2.threshold(img_ref_gray, 10, 255, cv2.THRESH_BINARY_INV)


imgCnts = cv2.findContours(img_ref_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgCnts = imutils.grab_contours(imgCnts)
imgCnts = imutils.contours.sort_contours(imgCnts, method="left-to-right")[0]
digits = {}

# cv.drawContours(img_ref, imgCnts, -1, (0,255,0), 2)


# 轮廓处理
for (i, c) in enumerate(imgCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = img_ref_gray[y:y+h, x:x+w]
    # cv_show(img_ref[y:y + h, x:x + w])
    roi = cv2.resize(roi, (57,88))
    #更新字典
    digits[i] = roi


 #核函数构建
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
rectKernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

img = cv2.imread('assets/test3.png')
img = imutils.resize(img, width=300)
img_resized = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 礼貌运算突出较亮的区域
tophat_img = cv2.morphologyEx(img_resized, cv2.MORPH_TOPHAT, rectKernel)

_,tophat_img = cv2.threshold(tophat_img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

gradX = cv2.Sobel(tophat_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(tophat_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

sobelx_abs = np.abs(gradX)
sobely_abs = np.abs(gradY)
#
# # 将绝对值转换回cv.CV_8U类型
sobelx_abs_uint8 = cv.convertScaleAbs(sobelx_abs)
sobely_abs_uint8 = cv.convertScaleAbs(sobely_abs)

sobel_combined = cv.addWeighted(sobelx_abs_uint8, 0.5, sobely_abs_uint8, 0.5, 0)


sobel_combined = cv2.morphologyEx(sobel_combined, cv2.MORPH_CLOSE, rectKernel)


_,sobel_combined = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


_,thresh = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, rectKernel,iterations=1)


thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, rectKernel5)

cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv.drawContours(img, cnts, -1, (0,255,0), 2)

locs = []

#筛选一下轮廓
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = w/float(h)

    if ratio>2.5 and ratio < 4.0:
        if(w>40 and w<55) and (h>10 and h<20):
            locs.append((x,y,w,h))


locs = sorted(locs,key=lambda x:x[0])

for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []

    group_color = img[gY -2 :gY +gH +2, gX -2:gX+gW +2]
    group = img_resized[gY -2 :gY +gH +2, gX -2:gX+gW +2]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    # cv_show(group)

    # #轮廓查找,排序

    digitCnts,_ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    cv.drawContours(group_color, digitCnts, -1, (0,255,0), 1)

    num_contours = len(digitCnts)
    print("找到的轮廓数量：", num_contours)

    # cv_show(group_color)

    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]

        roi = cv2.resize(roi, (57, 88))

        # cv_show(roi)

        scores = []
        #模板匹配
        for (digit, digitROI) in digits.items():

            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)

            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
            # print(scores)

        groupOutput.append(str(np.argmax(scores)))

        # print(groupOutput)

    cv2.rectangle(img, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)

    cv2.putText(img, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 255), 2)

    # print(groupOutput)

cv_show(img)


