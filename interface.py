from tkinter import *
from PIL import ImageGrab
import numpy as np
import NeuralNetwork as nn

def draw(event):
    x1, y1 = (event.x - 20), (event.y - 20)
    x2, y2 = (event.x + 20), (event.y + 20)
    c.create_oval(x1, y1, x2, y2, fill='black', outline='black')

def clear(event):
    c.delete("all")
    res.set(value='')

def recognize():
    x1 = root.winfo_rootx() + c.winfo_x()
    y1 = root.winfo_rooty() + c.winfo_y()
    x2 = x1 + c.winfo_width()
    y2 = y1 + c.winfo_height()
    img = ImageGrab.grab(bbox=(x1 + 2, y1 + 2, x2 - 4, y2 - 4))
    img = img.resize((28, 28)).convert('L')
    img = np.asarray(img, dtype=np.float32)
    img = 255 - img
    
    inputs = img.reshape(1, 784)    
    inputs = (inputs / 255.0 * 0.99) + 0.01
    outputs = nnet.query(inputs)
    label = np.argmax(outputs)
    res.set(value=str(label))

nnet = nn.NeuralNetwork()
nnet.load_weights("weights_input_hidden.npy", "weights_hidden_output.npy")
root = Tk()
root.title("Handwritten Digit Recognition")
root.geometry("1150x700+400+250")
root.resizable(False, False)
c = Canvas(root, bg='white', bd=1, highlightbackground='black')
c.place(x=50, y=50, width=500, height=500)
c.bind("<B1-Motion>", draw)
c.bind("<Button-3>", clear)
res = StringVar(root, value='')
Entry(root, borderwidth=3, textvariable=res, justify='center',
      font=("Helvetica", 300), state=DISABLED).place(x=600, y=50, width=500, height=500)
Button(root, bg="#E1E1E1", font=("BankGothic Md BT", 38), text="Recognize",
       command=recognize).place(x=50, y=583, width=500, height=80)
root.mainloop()

