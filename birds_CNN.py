from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import keras
import numpy as np


classes = ["barn_swallow","crow","great_tit","japanese_white_eye","pigeon","pygmy_woodpecker","sparrow","redstart"]
num_classes = len(classes)
epochs=50
batch_size=32
model = Sequential()
#メインの関数を定義する
def main():
    X_train, X_test, y_train, y_test = np.load("./birds_aug.npy") #保存したデータをロードする
    X_train = X_train.astype("float") / 255 #X_trainの値は0~255だがそれを255で割ることですべての値を0~1にする(正規化)
    X_test = X_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, num_classes) #正解ラベルをnumpy用に変換する
    y_test = np_utils.to_categorical(y_test, num_classes) #第一引数は正解は何番目か、第二引数は全体のサイズ

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

def model_train(X, y):
    medel = Sequential()
    model.add(Conv2D(32,(3,3),padding="same",input_shape=X.shape[1:]))
    #32:畳み込むフィルターの数,　(3,3):フィルターのサイズ,　padding="same":畳み込むとデータが小さくなる(次元が下がる)が、
    #そうならないように畳み込む前のデータの周りに0を付け足して、畳み込み後にもとの大きさにちょうど戻るようにする,
    #X_train.shape>>>[497,50,50,3]:50*50のRGB3色分のデータが497個ある。input_shapeはX_train.shape[1:]とすればおｋ
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3))) #少し小さくなるはず
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.9))

    #model.add(Conv2D(32, (3, 3), padding='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    #model.add(Conv2D(64, (3, 3), padding='same'))
    #model.add(Activation('relu'))
    #model.add(Conv2D(64, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten()) #全結合層に入れる前に一次元に変換しておく
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
     #epochs=10,testacc=0.53625,testloss=2.105,
     #epochs=25,testacc=0.47,testloss=1.7.,D=0.9  １層
     #epochs=25,testacc=0.43,testloss=3.4,D=0.25　１層　過学習
     #epochs=10,testacc=0.40,testloss=2.8,D=0.25 一層　
     #epochs=25,testacc=0.48,testloss=1.47,D=0.95 一層
     #epochs=10,testacc=0.43,testloss=1.63,D=0.25,0.25 2層
     #epochs=10,testacc=0.35,testloss=1.68,D=0.5,0.5 2層
     #epochs=10,testacc=0.22,testloss=1.9,D=0.8,0,8
     #epochs=10,testacc=0.47,testloss=1.5,D=0.25,0.25,0.25
     #epochs=50,testacc=0.,testloss=0.,D=0.25,0.25,0.25 aug
     #epochs=25,testacc=0.65,testloss=1.063,D=0.9 aug  epochs=50でやってみる
     #epochs=25,testacc=0.49,testloss=2.527,D=0.25,0.25
     #epochs=25,testacc=0.,testloss=0.,D=0.25,0.25 aug 重すぎて無理
     #epochs=25,testacc=0.,testloss=0.,D=0.25,0.25 aug:i>=200
     #epochs=50,testacc=0.49,testloss=3.053,D=0.25,0.25  二層あると過学習してしまう?
     #epochs=25,testacc=0.445,testloss=1.56,D=0.95
     #epochs=75,testacc=0.49,testloss=1.8,D=0.9
     #epochs=50,testacc=0.64,testloss=1.47,D=0.9 aug 終了予想時間14:20
     #epochs=100,testacc=0.,testloss=0.,D=0.9






    model.compile(loss='categorical_crossentropy', #損失関数　よくわからん
              optimizer=opt, #最適化関数
              metrics=['accuracy']) #評価方法:この場合は正解率,ex)精度,再現率,F値

    model.fit(X, y, batch_size=batch_size, epochs=epochs) #学習の実行

    model.save("./birds_CNN.h5") #モデルを保存する

    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1) #verbose:バーバスと読むらしい。1にすると途中経過を表示するらしい。
    print('Test loss: ', scores[0])
    print('Test accuracy: ', scores[1])

if __name__ == "__main__":
    main()
