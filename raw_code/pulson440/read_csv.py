import numpy as np
import pandas
import pickle

def read_csv(filename):
    to_return = pandas.read_csv(filename)
    print(to_return)
    return to_return

read_csv("ExampleMotionCapture.csv")

with open("Mandrill_1way_Misaligned_data.pkl", "rb") as file:
    data = pickle.load(file)
