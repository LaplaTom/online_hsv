#encoding=utf-8
import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES']='4'
from flask import Flask, render_template, request, url_for,session

import torch
import cv2
import numpy as np
import re
import base64
from PIL import Image
import matplotlib.pyplot as plt
import idn.online2offline as online2offline
import idn.idn_test as idn_test
import lcq_idn.lcq_idn_test as lcq_idn_test
from datetime import timedelta

# 1. 初始化 flask app
# 强制性的
app = Flask(__name__)
app.config['DEBUG']=True
app.config['SEND_FILE_MAX_AGE_DEFAULT']=timedelta(seconds=1)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
# 2. 初始化global variables

# 添加数据到session中
# 操作session的时候，跟操作字典是一样的
# SECRET_KEY
 
@app.route('/set/',methods=['GET', 'POST'])
def set():
    name = request.get_data().decode()
    session['username'] = name
    # 如果没有指定session的过期时间，那么默认是浏览器关闭后就自动结束
    # 如果设置了session的permanent属性为True，那么过期时间是31天。
    session.permanent = True
    return 'seccess'
 
@app.route('/get/')
def get():
    # sesssion['username']  # 若不存在是会报错，建议使用get()
    # session.get('username')
    print(session.get('username'))
    if session.get('username') == None:
       return 'error'
    else:
      img_num = 0
      img_file_path = './cache/'+session.get('username')
      if os.path.exists(img_file_path):
          img_num = len(os.listdir(img_file_path))
      else:
         os.mkdir(img_file_path)
      return [session.get('username'),str(img_num)]
 
@app.route('/delete/')
def delete():
  #  print(session.get('username'))
    print('delete session:',session.get('username'))
    session.pop('username')
    return 'success'

@app.route('/delete_all/')
def delete_all():
  #  print(session.get('username'))
    d_a_name = session.get('username')
    print('session:',session)
    session.pop('username')
  #  print(session.get('username'))
    d_a_path = './cache/'+str(d_a_name)
    d_a_path_txt = './cache/'+str(d_a_name)+'_txt'
    if os.path.exists(d_a_path):
        shutil.rmtree(d_a_path)
    if os.path.exists(d_a_path_txt):
        shutil.rmtree(d_a_path_txt)
    return 'success'
 

 
@app.route('/clear/')
def clear():
    print(session.get('username'))
    # 删除session中的所有数据
    session.clear()
    print(session.get('username'))
    return 'success' 

# 3. 将用户画的图输出成output.png
def convertImage(imgData1):
  imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
  with open('output.png', 'wb') as output:
    output.write(base64.b64decode(imgstr))

# 4. 搭建前端框架
# 装饰器，那个url调用相关的函数
@app.route('/')
def index():
  return render_template("index.html")

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
    if filename:
        file_path = os.path.join(app.root_path, endpoint, filename)
        values['q'] = int(os.stat(file_path).st_mtime)
        return url_for(endpoint, **values)

def save_img(name,img):
    img_file_path = './cache/'+str(name)
    img_num = len(os.listdir(img_file_path))
    cv2.imwrite(os.path.join(img_file_path,str(img_num+1)+'.jpg'),img)
    return os.path.join(img_file_path,str(img_num+1)+'.jpg')

def save_json(name,point):
    img_file_path = './cache/'+str(name)+'_txt'
    if os.path.exists(img_file_path):
          img_num = len(os.listdir(img_file_path))
    else:
         os.mkdir(img_file_path)
    img_num = len(os.listdir(img_file_path))
    wj=open(os.path.join(img_file_path,str(img_num+1)+'.txt'),'w')
    for line in point:
        wj.write(str(line[0])+' '+str(line[1])+' '+str(line[2])+'\n')
    wj.close()

def get_test_data(name,test_img):
    con_img = []
 #   test_img = cv2.imread(img_path, 0)
    test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])
    for i in range(1,6):
     #   print('test:',i)
        refer_img = cv2.imread('./cache/'+str(name)+'/'+str(i)+'.jpg', 0)
        refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])
        refer_test = np.concatenate((refer_img, test_img), axis=0)
        con_img.append(torch.FloatTensor(refer_test))
    return con_img

# 5. 定义预测函数
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
  # 这个函数会在用户点击‘predict’按钮时触发
  # 会将输出的output.png放入模型中进行预测
  # 同时在页面上输出预测结果
  point_data = []
  imgData = request.get_data().decode().split("??lcq??")
  imgname = imgData[0]
  imdata = imgData[1].split(",") 
 # print(imgname,imdata)
  for i in range(0,len(imdata),3):
     #print(i,imgData[i],len(imgData))
     point_data.append([imdata[i],imdata[i+1],imdata[i+2]])
  save_json(imgname,point_data)
  im = online2offline.read_from_strokes(online2offline.xyz(point_data))
 # save_img_path = save_img(imgname,im)
  save_img(imgname,im)
  if len(os.listdir('./cache/'+str(imgname)))<6:
     return 'r'
  
#  acc = idn_test.model_run(get_test_data(imgname,im))
  acc = lcq_idn_test.model_run(get_test_data(imgname,im))
  if(acc >= 0.6):
     return 'g'
  else:
     return 'f'
  
# 6. 返回本地访问地址
if __name__ == "__main__":
    # 让app在本地运行，定义了host和port
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=5050)