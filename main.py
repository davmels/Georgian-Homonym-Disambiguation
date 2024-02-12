from models.Transformers_TC import fine_tune_transformer_TC
from models.Transformers_FM import fine_tune_transformer_FM
from models.LSTM import train_LSTM

if __name__ == '__main__':
    '''decide between which model you want to train'''
    fine_tune_transformer_FM()
    #fine_tune_transformer_TC()
    #train_LSTM(show=True)