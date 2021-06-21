
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import * 
from tkinter import messagebox

x_train = pd.read_csv('Diabetes_XTrain.csv')
y_train = pd.read_csv('Diabetes_YTrain.csv')

X = x_train.values
Y = y_train.values



def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(X,Y,querypoint,k=15):
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(querypoint,X[i])
        vals.append((d,Y[i]))
    
    vals = sorted(vals)
    vals = vals[:k]
    
    vals = np.array(vals)
    #print(vals)
    new_vals = np.unique(vals[:,1],return_counts = True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred
    
def prediction():
    preganancies = float(E1.get())
    Glucose = float(E2.get())
    BloodPressure = float(E3.get())
    SkinThickness = float(E4.get())
    Insulin = float(E5.get())
    BMI = float(E6.get())
    DiabetesPredigreeFunction = float(E7.get())
    Age = float(E8.get())
    l = [preganancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPredigreeFunction,Age]
    number = knn(X,Y,l)
    if number==1:
        messagebox.showinfo("Prediction","Yes you have chances to be diabetic postive")
    else:
        messagebox.showinfo("Prediction","No you dont have chances for diabetics")

root = Tk()

root.title("Diabetes Prediction")
Label(root,text="Pregnancies").grid(row=0)
Label(root,text="Glucose").grid(row=1)
Label(root,text="BloodPressure").grid(row=2)
Label(root,text="SkinThickness").grid(row=3)
Label(root,text="Insulin").grid(row=4)
Label(root,text="BMI").grid(row=5)
Label(root,text="DiabetesPedigreeFunction").grid(row=6)
Label(root,text="Age").grid(row=7)
E1 = Entry(root,borderwidth = 5)
E1.grid(row=0,column=2)
E2 = Entry(root,borderwidth = 5)
E2.grid(row=1,column=2)
E3 = Entry(root,borderwidth = 5)
E3.grid(row=2,column=2)
E4 = Entry(root,borderwidth = 5)
E4.grid(row=3,column=2)
E5 = Entry(root,borderwidth = 5)
E5.grid(row=4,column=2)
E6 = Entry(root,borderwidth = 5)
E6.grid(row=5,column=2)
E7 = Entry(root,borderwidth = 5)
E7.grid(row=6,column=2)
E8 = Entry(root,borderwidth = 5)
E8.grid(row=7,column=2)



button = Button(root,text="Predict Diabetes",width = 20, bg= "Blue",fg = "white",command = prediction).grid(row=8,column = 1)


root.mainloop()


