from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

def main():
 AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
    }

 df_AND = pd.DataFrame(AND)


 X,y = prepare_data(df_AND)

 ETA = 0.3
 EPOCHS = 10


 model_AND = Perceptron(eta = ETA, epochs = EPOCHS)
 model_AND.fit(X, y)
 _ = model_AND.total_loss()

 save_model(model_AND,"and.model")

 save_plot( df_AND, "and.png", model_AND)

if __name__ == "__main__":
 main()