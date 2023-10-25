from tkinter import filedialog
from tkinter import *

# build form
frmMain = Tk()

Tb1 = Text(frmMain, bd=10, height=12)
Tb1.grid(row=0, column=0, padx=10, pady=10, sticky=W + E)

##frmMain.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("CSV files","*.csv"),("all files","*.*")))
##NameOfFile = frmMain.filename
##f = open(NameOfFile,"r")

##f.close


frmMain.mainloop()