import os
from flask import Flask, request, redirect, url_for, render_template
from flask import flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
import tensorflow as tf
from PIL import Image
import keras,sys
import numpy as np

classes = ["barn_swallow","crow","great_tit","japanese_white_eye","pigeon","pygmy_woodpecker","sparrow","redstart"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./birds_CNN.h5')#add
graph = tf.get_default_graph()#add

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph#add
    with graph.as_default():#add
        if request.method == 'POST':#add indent
            if 'file' not in request.files:#add indent
                flash('ファイルがありません')#add indent
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            #model = load_model("./animal_CNN.h5")

                image = Image.open(filepath) #コマンドプロンプトの第二引数からファイルを受け取る
                image = image.convert("RGB") #いろいろ変換
                image = image.resize((image_size,image_size))
                data = np.asarray(image)
                X = []
                X.append(data)
                #X = x #なんか無駄なことしてる
                X = np.array(X)

                result = model.predict([X])[0]
                predicted = result.argmax()
                percentage = int(result[predicted]*100)

                return ("ラベル: " + classes[predicted] + ", 確率: " + str(percentage) + " %")

                #return redirect(url_for('uploaded_file',filename=filename))

        return render_template("index.html")

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


if __name__ == '__main__':
    app.run()