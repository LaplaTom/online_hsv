import numpy as np
import cv2
import os

def read_from_strokes(strokes,margin=10,color=(0,0,0),bgcolor=(255,255),thickness=2):
    ''' 将联机手写笔划数据转成图片
    :param strokes: n个笔划,每个笔划包含不一定要一样长的m个点，每个点是(x，y)的结构
    :param margin:图片边缘
    :param color:前景笔划颜色,默认黑色: param bgcolon:背录颜色，默认白色
    : param thickness:笔划粗度
    '''

    
  #  print(strokes)
    #1边界
    minx = min([p[0] for s in strokes for p in s])
    miny = min([p[1] for s in strokes for p in s])
    for stroke in strokes:
        for p in stroke:
            p[0] -= minx - margin
            p[1] -= miny - margin
    maxx= max([p[0] for s in strokes for p in s])
    maxy = max([p[1] for s in strokes for p in s])
    #2画出图片
    canvas = np.ones((140,280),dtype=np.uint8)*255
    #画出每个笔划轨迹
    for stroke in strokes:
        for i in range(len(stroke) - 1):
            cv2.line(canvas,stroke[i], stroke[i + 1],color, thickness=thickness)
       #     print("12")
    return canvas

def xyz(data):
    strokes = []
    strokes_line = []
    max_x = 0
    max_y = 0
    rate = 0
    for line in data:
        if (int(float(line[2].strip(' ')))) == 2:
            strokes.append(strokes_line)  
            strokes_line = []
        if (int(float(line[2].strip(' ')))) == 1:
            strokes_line = []
        if  (int(float(line[2].strip(' ')))) == 0:
            wrx = int(float(line[0].strip(' ')))
            wry = int(float(line[1].strip(' ')))
            if(wrx > max_x):
                max_x = wrx
            if(wry > max_y):
                max_y = wry
            #  print(wrx,wry)
            strokes_line.append([wrx,wry])
        
     #   print(data_line)
    if max_x/256 > max_y/120:
        rate = float(max_x/256)
    else:
        rate = float(max_y/120)
    print(rate)
    for x in range(len(strokes)):
        for y in range(len(strokes[x])):
            for z in range(len(strokes[x][y])):
                strokes[x][y][z] = int(strokes[x][y][z]//rate)

    return strokes


def txt2xy(file_path):
    strokes = []
    file2 = open(file_path,'r')
    strokes_line = []
    flag = 1
    for line in file2:
        if flag == 1:
            flag =0
            continue
        while line.find("  ")+1:
            line = line.replace("  "," ")
        #print(line)
        data_line = line.lstrip(' ').strip('\n').split(" ")  # 去除首尾换行符，并按空格划分
     #   print(data_line)
        if (int(float(data_line[4].strip(' ')))) == 1:
            strokes_line = []
        if (int(float(data_line[4].strip(' ')))) == 0:
            wrx = int(float(data_line[0].strip(' ')))
            wry = int(float(data_line[1].strip(' ')))
          #  print(wrx,wry)
            strokes_line.append([wrx,wry])
        if (int(float(data_line[4].strip(' ')))) == 2:
            strokes.append(strokes_line)  
     #   print(data_line)
    file2.close()
    return strokes

 
def mid(li):#构造函数对于列表进行处理    
   
    if len(li)%2==0: #元素个数为偶数
        b=len(li)//2-1
        c=len(li)//2
        return (li[b]+li[c])/2
        
    elif len(li)%2!=0:  #元素个数奇数
        a=len(li)//2
        return li[a]
 

 

if __name__ == '__main__':
  #  file_path = 'f_10_3.txt'
    files_path = r'./'
    im_x = []
    im_y = []
    for root, dirs, files in os.walk(files_path):
        for file in files:
            path = os.path.join(root, file)
            #print(file)
        
    #         g_99_9.png
    #         F:\LCQ\研究生活\A实验室的啊啊啊啊啊啊啊\数据集\笔记认证\MSDS\MSDS-TDS\session2\99\images\g_99_9.png
    #         F:\LCQ\研究生活\A实验室的啊啊啊啊啊啊啊\数据集\笔记认证\MSDS\MSDS-TDS\session2\99\images\g_99_9.png
            if file.find(".txt")+1:
            #  path = r'MSDS-ChS-tiny\session1\14\images\f_14_2.jpg'
                print(path)
                im = read_from_strokes(txt2xy(path))
                # # if os.path.exists(os.path.split(new_path)[0]):
                # #     print(os.path.split(new_path)[0])
                # #     os.mkdir(os.path.split(new_path)[0])
                # cv2.imwrite(new_path,255-a[:,:,3])
                # a = cv2.resize(a, (256, 128))
                path = path.replace("series","images").replace("txt","jpg")
            #  os.remove(path)
              #  print(path)
                im_x.append(im.shape[0])
                im_y.append(im.shape[1])
                cv2.imwrite(path,im)
    
    if (len(im_x) == len(im_y)):
        print(f'总共:{len(im_x)}')
        print(f'mean_x:{sum(im_x)/len(im_x)},mid_x:{mid(sorted(im_x))}')
        print(f'mean_y:{sum(im_y)/len(im_y)},mid_y:{mid(sorted(im_y))}')
   # cv2.imwrite("123.jpg",im)