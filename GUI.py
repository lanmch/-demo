from tkinter import *
import os
def open_point_search() :
    os.system('start Project3.exe')
def open_video_demo():
    os.system('start python video_demo.py')

def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    print(size)
    root.geometry(size)
root = Tk()
root.title('冷刃工作室')
center_window(root,600,400)
root.minsize(600,400)
b1 = Button(root,text = "关键点检测",command = open_point_search)
b1['width'] = 30
b1['height'] = 3
b1.pack(expand = YES)
b2 = Button(root,text = "情绪分析",command = open_video_demo)
b2['width'] = 30
b2['height'] = 3
b2.pack(expand = YES)
root.mainloop()