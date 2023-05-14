# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import os
import json
import time
import web
import numpy as np
import uuid
from PIL import Image
import os
import tensorflow as tf
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import importlib, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
web.config.debug  = True

filelock='file.lock'
if os.path.exists(filelock):
   os.remove(filelock)

render = web.template.render('templates', base='base')
from config import *
from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL
from application import trainTicket,idcard 
if yoloTextFlag =='keras' or AngleModelFlag=='tf' or ocrFlag=='keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K
        #tf.disable_eager_execution()
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3## GPU最大占用量
        config.gpu_options.allow_growth = True##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())
    
    else:
      ##CPU启动
      os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag=='opencv':
    scale,maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag=='darknet':
    scale,maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag=='keras':
    scale,maxScale = IMGSIZE[0],2048
    from text.keras_detect import  text_detect
else:
     print( "err,text engine in keras\opencv\darknet")
     
from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase
    ocr = redisDataBase().put_values
else:   
    from crnn.keys import alphabetChinese,alphabetEnglish
    if ocrFlag=='keras':
        from crnn.network_keras import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
            
    elif ocrFlag=='torch':
        from crnn.network_torch import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense
                
        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag=='opencv':
        from crnn.network_dnn import CRNN
        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print( "err,ocr engine in keras\opencv\darknet")
     
    nclass = len(alphabet)+1   
    if ocrFlag=='opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")
        
    ocr = crnn.predict_job
    
   
from main import TextOcrModel# 接main.py

model =  TextOcrModel(ocr,text_detect,angle_detect)
    

billList = ['通用OCR']

class OCR:

    def GET(self):
        post = {}
        post['postName'] = 'ocr'##请求地址
        post['height'] = 1000
        post['H'] = 1000
        post['width'] = 600
        post['W'] = 600
        post['billList'] = billList
        return render.ocr(post)

    def POST(self):
        t = time.time()
        data = web.data()
        uidJob = uuid.uuid1().__str__()
        
        data = json.loads(data)
        billModel = data.get('billModel','')
        textAngle = data.get('textAngle',False)##文字检测
        textLine = data.get('textLine',False)##只进行单行识别
        
        imgString = data['imgString'].encode().split(b';base64,')[-1]
        img = base64_to_PIL(imgString)# base64字符串转换为PIL图像对象
        if img is not None:
            img = np.array(img)
            
        H,W = img.shape[:2]

        while time.time()-t<=TIMEOUT:
            if os.path.exists(filelock):
                continue
            else:
                with open(filelock,'w') as f:
                    f.write(uidJob)
                partImg = Image.fromarray(img)
                text    = crnn.predict(partImg.convert('L'))
                res =[ {'text':text,'name':'0','box':[0,0,W,0,W,H,0,H]} ]
                os.remove(filelock)
                break

        if text != "":
            tf.flags.DEFINE_string("open_data_file", "./data/open.txt", "Data source for the open data.")
            tf.flags.DEFINE_string("close_data_file", "./data/close.txt", "Data source for the close data.")
            tf.flags.DEFINE_string("lure_data_file", "./data/lure.txt", "Data source for the lure data.")

            # Eval Parameters
            tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
            # 填写训练获得模型的存储位置
            tf.flags.DEFINE_string("checkpoint_dir", "./runs/1683533520/checkpoints/",
                                   "Checkpoint directory from training run")
            tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

            # Misc Parameters
            tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
            tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
            FLAGS = tf.flags.FLAGS

            if FLAGS.eval_train:
                pass
            else:
                x1 = text
                x_raw = [x1]
                y_test = [0, 0, 1]
                print(x1)

            # Map data into vocabulary
            vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
            vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
            x_test = np.array(list(vocab_processor.transform(x_raw)))

            # Evaluation
            # ==================================================
            checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

            graph = tf.Graph()
            with graph.as_default():
                session_conf = tf.ConfigProto(
                    allow_soft_placement=FLAGS.allow_soft_placement,
                    log_device_placement=FLAGS.log_device_placement)
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    # Load the saved meta graph and restore variables
                    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

                    saver.restore(sess, checkpoint_file)

                    # Get the placeholders from the graph by name
                    input_x = graph.get_operation_by_name("input_x").outputs[0]
                    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                    # Tensors we want to evaluate
                    predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                    # Generate batches for one epoch
                    batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

                    # 存储模型预测结果
                    all_predictions = []
                    for x_test_batch in batches:
                        batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                        all_predictions = np.concatenate([all_predictions, batch_predictions])

            # Print accuracy if y_test is defined
            if y_test is not None:
                correct_predictions = float(sum(all_predictions == y_test))

            y = []
            for i in all_predictions:
                if i == 0.0:
                    print("明确打开")
                elif i == 1.0:
                    print("明确关闭")
                else:
                    print("诱导性文字")

        timeTake = time.time()-t
        return json.dumps({'res':res,'timeTake':round(timeTake,4)},ensure_ascii=False)
        

urls = ('/ocr','OCR',)

if __name__ == "__main__":

      app = web.application(urls, globals())
      app.run()
