import os,cv2
import base64
import numpy as np
from flask import  render_template,Flask,request
from keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
from tqdm.notebook import tqdm
import tensorflow as tf
age_model = load_model('Agemodel.h5')
gender_model = load_model('Gendermodel.h5')
app=Flask(__name__)


@app.route('/',methods=['GET'])
def home():

    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        image=request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'New_Data', secure_filename(image.filename))
        image.save(file_path)
        img = tf.keras.utils.load_img(file_path, grayscale=True)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img)

        g,a = Result(img)
        result = class_name(g,a)
        # img=display(value)

        return result,file_path
    return None


def class_name(g,a):
    age="Age Range :{}".format(a)
    if (g==1):
        gender="Gender : Female"

    else:
        gender = "Gender : Male"
        return "Result: ."+gender+" ."+age
def Result(img):
    ranges = [' 1-2', ' 3-9', ' 10-20', ' 21-27', ' 28-45', ' 46-65', ' 66-116']
    gender_dict = {0: 'Male', 1: 'Female'}

    pred = age_model.predict(img.reshape(1, 128, 128, 1))
    pred_age = np.argmax(pred)
    age = ranges[pred_age]
    gender_pred = gender_model.predict(img.reshape(1, 128, 128, 1))
    pred_gender = gender_dict[int(gender_pred[0][0])]
    gender = gender_dict[int(pred[0][0])]

    return (gender,age)





if __name__=="__main__":
    app.run(debug=True)