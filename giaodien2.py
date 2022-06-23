from cProfile       import label
from pickle         import FROZENSET
from tkinter        import *
from tkinter        import filedialog
from tkinter.ttk import Frame, Button, Style
from PIL import  Image,ImageTk
import os
import tkinter as tk
from PIL                import  Image,ImageTk
from keras.models import load_model
import matplotlib.pyplot as plt

import cv2
import numpy as np

model = load_model('C:\Ols\Document\AI\web\Breast_cancer.h5')
classes = ['Benign Case','Malignant Case' ,'Normal Case']


def showimage():
    global fln
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("PNG File","*.png"),("JPG File","*.jpg"), ("ALL Files","*.*")))
    img = Image.open(fln)
    img.thumbnail((300,300))
    img = ImageTk.PhotoImage(img)
    lbl6.configure(image= img)
    lbl6.image = img

def recognize():
    global lbl1
    global lbl2
    global lbl3
    global lbl4
    global lbl5
    img_path= fln
    img=plt.imread(img_path)
    print ('Input image shape is ', img.shape)
    img=cv2.resize(img, (200,200)) 
    print ('the resized image has shape ', img.shape)
    plt.axis('off')
    plt.imshow(img)
   
    img=np.expand_dims(img, axis=0)
    print ('image shape after expanding dimensions is ',img.shape)
    pred=model.predict(img)
    print ('the shape of prediction is ', pred.shape)
    index=np.argmax(pred[0])
    klass=classes[index]
    probability=pred[0][index]*100
    print(f'the image is predicted as being {klass} with a probability of {probability:6.2f} %')
    lbl1 = Label(root,text = f"DETECT: {klass}" , fg= "darkolivegreen", font=("Arial", 20))
    lbl1.pack(pady= 20)
    lbl2 = Label(root,text = f"ACCURACY: {probability:6.2f} %" , fg= "teal", font=("Arial", 20))
    lbl2.pack(pady= 20)
    if(index == 2):
        lbl3 = Label(root,text ="Xin chúc mừng! Cơ thể bạn chẳng có khối u nào cả!!!", fg= "brown", font=("Arial", 20))
        lbl3.pack(pady= 20)
    elif(index == 1):
        lbl3 = Label(root,text ="Thật buồn nhưng phải thông báo là bạn đã có một khối u ác tính :(((", fg= "seagreen", font=("Arial", 20))
        lbl3.pack(pady= 20)
        lbl4 = Label(root,text ="Đừng bi quan vì kết quả bạn nhé!", fg= "seagreen", font=("Arial", 20))
        lbl4.pack(pady= 20)
        lbl5 = Label(root,text ="Hãy để bác sĩ tư vấn giúp bạn loại bỏ khối u xấu tính này nhé <33", fg= "seagreen", font=("Arial", 20))
        lbl5.pack(pady= 20)
    elif(index == 0):
        lbl3 = Label(root,text ="Đây chỉ là một khối u lành tính ", fg= "magenta", font=("Arial", 20))
        lbl3.pack(pady= 20)
        lbl4 = Label(root,text ="Nhưng mà bạn không được chủ quan đâu đấy nhé", fg= "magenta", font=("Arial", 20))
        lbl4.pack(pady= 20)
        lbl5 = Label(root,text ="Bạn hãy làm theo các hướng dẫn của bác sĩ và chăm sóc bản thân thật nhiều nhé!!", fg= "magenta", font=("Arial", 20))
        lbl5.pack(pady= 20)
    return

def clear():
    lbl1.after(1000, lbl1.destroy())
    lbl2.after(1000, lbl2.destroy())
    lbl3.after(1000, lbl3.destroy())
    lbl4.after(1000, lbl4.destroy())
    lbl5.after(1000, lbl5.destroy())
    return

root = Tk()
root.title("Detect Breast Cancer")
root.geometry("1024x800")
# lbl1 = Label(root)
# lbl2 = Label(root)

frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl = Label(root,
            text = "DETECT BREAST CANCER", 
            fg= "white", 
            font=("Arial", 30), 
            background="gray"
            )
lbl.pack(padx=10, pady= 10)

lbl6 = Label(root)
lbl6.pack()

root.style = Style()
root.style.theme_use("clam")

btn = Button(frm,text = "Browser Image", command= showimage)
btn.pack(side=tk.LEFT,padx= 30)

btn1 = Button(frm,text = "Recognize", command= recognize)
btn1.pack(side=tk.LEFT,padx= 30, pady = 10)

btn3 = Button(frm,text = "Clear", command= clear)
btn3.pack(side=tk.LEFT,padx=30)

btn2 = Button(frm,text = "Exit",command=lambda: exit())
btn2.pack(side=tk.LEFT,padx=30)

root.mainloop()
