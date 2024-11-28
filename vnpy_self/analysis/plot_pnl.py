import pandas as pd
from pt_tools import *
import sys

def run(file_name):
    d = pd.read_csv(f'../{file_name}')
    print(d.tail())
    return plot(1000000, d.sum(1))

if __name__ == "__main__":
    file_name = sys.argv[1]
    run(file_name)