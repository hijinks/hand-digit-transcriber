import cv2
import PIL
from PIL import ImageTk
import os
import Tkinter as tk
from Tkinter import *
import threading
import time
from threading import Lock
lock = Lock()

class image_prompt():

    def __init__(self, image):
        self.root = Tk()
        self.response = False
        self.canvas = Canvas(self.root, width=100, height=100)
        self.label = Label(self.root, text='Response')
        self.entry = Entry(self.root)
        self.entry.bind('<Key-Return>', self.response)
        self.entry.pack()
        self.entry.focus_set()
        self.canvas.pack()
        im = PIL.Image.fromarray(image)
        tk_img = ImageTk.PhotoImage(im)
        self.canvas.create_image(250, 250, image=tk_img)

    def response(self):
        print 'yo'
        self.response = self.entry.get()
        self.root.destroy()

class App(Frame):
    def __init__(self, image):
        self.root = Tk()
        self.output = False
        self.entry = Entry(self.root)
        self.entry.pack()
        self.canvas = Canvas(self.root, width=300, height=300)
        self.canvas.pack()
        self.canvas.create_image(100, 100, image=image)
        self.entry.bind('<Key-Return>', self.response)
        self.root.mainloop()

    def response(self, event):
        self.output = self.entry.get()
        self.root.destroy()


im = cv2.imread('photo_5.jpg')
iP = App(im)
print iP.output