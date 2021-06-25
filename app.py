from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import statistics
app = Flask(__name__)

@app.route('/')  
def upload():  
    return render_template("inputFile.html")

@app.route('/about')  
def about():  
    return render_template("aboutUs.html")

@app.route('/output', methods = ['POST'])
def fungiClassification():
    if request.method == 'POST':
        labels = ['C. auris','C. krusei', 'C. tropicalis']
        img_crops=[]
        f = request.files['image']  
        f.save('static/'+f.filename)  

        # Reading the image file
        M=150
        N=150
        IMG_SHAPE = (224, 224, 1)
        img = cv2.imread('static/'+f.filename)
        #img = load_img(fn)
        img = np.array(img, dtype='uint8')
        #cv2.imshow('img',img)
        # Convert into grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_img = np.array(gray_img, dtype='uint8')
        
        crops = [gray_img[x:x+M,y:y+N] for x in range(0,img.shape[0]-M+1,M) for y in range(0,img.shape[1]-N+1,N)]
        print("No. of crops is: ",len(crops))
        i=0
        num=0
        for crop in crops:
            num=num+1
            # Convert into grayscale
            #gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            print("Non zero pixels: ",cv2.countNonZero(crop))
            
            #crop_name='img_crops_all/'+f.filename.replace('.*','_'+str('{:03}'.format(num))+'.jpg')
            #print(crop_name)
            #cv2.imwrite(crop_name, crop)
            mask = np.ones(crop.shape[0:2], dtype="uint8")
            mask[crop <= 20] = 0
            if (cv2.countNonZero(mask) >=80/100*N*M):
                print("true")
                crop = cv2.resize(crop, (224, 224))
                crop = ((crop-np.min(crop))/(np.max(crop)-np.min(crop)))*255
                crop = crop.astype(np.uint8)
                #i=i+1
                #temp=crop
                #cv2.imwrite('img_crops/'+f.filename.replace('.*','_'+str('{:03}'.format(num))+'.jpg'), crop)
                #crop = crop.astype(np.uint8)
                #print(crop.dtype)
                
                img_crops.append(crop)
                #crop = crop.astype('float32')
                #cv2.imshow('crop',crop)
                #cv2.waitKey(0)
        model = load_model('Models/fungi_classifier_model5_DA.h5')
        img_crops=np.array(img_crops)
        img_crops = img_crops.reshape((img_crops.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], 1))
        pred_prob = model.predict(img_crops)
        pred_prob=pred_prob*100
        np.set_printoptions(formatter={'float_kind':"{:.2f}".format})
        print("pred probabilities: ",pred_prob)
        #pred_class = model.predict_classes(x_test)
        pred_class = np.argmax(pred_prob, axis=-1)
        print("pred class: ",pred_class)

        final_class = statistics.mode(pred_class)
        print(final_class)
        final_class_name = labels[final_class]
        avg_prob = np.mean(pred_prob,axis=0)
        argsort_avg_prob=np.argsort(avg_prob)


        fig='/static/'+f.filename
    return render_template("output.html", final_class_name=final_class_name, fig = fig, labels=labels,avg_prob=avg_prob,argsort_avg_prob=argsort_avg_prob)

if __name__ == '__main__':
    app.run()