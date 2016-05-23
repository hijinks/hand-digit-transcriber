import numpy as np
import PIL
from PIL import ImageTk
from Tkinter import *
import random
import math

data_list = np.load('recent_hog_list.npy')
label_list = np.load('recent_label_list.npy')

randomIndex = math.floor(len(data_list)*random.random())

print(type(data_list))
print(len(label_list))

imageData = data_list[randomIndex]

np.savetxt("foo.csv", data_list, delimiter=",", fmt='%d')

print imageData

class imageDisplay(Frame):

    def __init__(self, parent, imageData):
        Frame.__init__(self, parent, background="white")

        self.parent = parent
        self.initUI(imageData)


    def initUI(self, imageData):

        self.parent.title("Simple")
        self.canvas = Canvas(self.parent, width=200, height=200)
        self.canvas.pack()
        im = PIL.Image.fromarray(imageData)
        tk_img = ImageTk.PhotoImage(im)
        self.canvas.create_image(0, 0, image=tk_img, anchor='nw')
        self.pack(fill=BOTH, expand=1)

def main():

    root = Tk()
    root.geometry("250x250+300+300")
    app = imageDisplay(root, imageData)
    root.mainloop()


if __name__ == '__main__':
    main()