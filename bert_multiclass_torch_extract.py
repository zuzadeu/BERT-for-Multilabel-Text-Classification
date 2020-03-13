from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser
from bert_serving.server.helper import get_shutdown_parser
from sklearn.metrics import f1_score
from bert_serving.client import BertClient
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import json
import os
import tensorflow as tf
from keras import backend as K
import collections
import numpy as np
from sacred import Experiment
#from sacred.observers import MongoObserver
ex = Experiment()
#ex.observers.append(MongoObserver(
    #url='mongodb://mongo_user:mongo_password@localhost:27017/?authMechanism=SCRAM-SHA-1'))

base_path = 'C:/Users/zuzan/Bert-Multi-label'

def get_columns_names():
    meta = pd.read_csv(os.path.join(
    base_path, "movie.metadata.tsv"), sep='\t', header=None)
    meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
    genres = meta[["movie_id", "movie_name", "genre"]]
    
    plots = pd.read_csv(os.path.join(
        base_path, "plot_summaries.txt"), sep='\t', header=None)
    plots.columns = ["movie_id", "plot"]
    
    genres['movie_id'] = genres['movie_id'].astype(str)
    plots['movie_id'] = plots['movie_id'].astype(str)
    movies = pd.merge(plots, genres, on='movie_id')
    
    genres_lists = []
    for i in movies['genre']:
        genres_lists.append(list(json.loads(i).values()))
    movies['genre'] = genres_lists
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit_transform(movies['genre'])
    # transform target variable
    y = multilabel_binarizer.transform(movies['genre'])
    for idx, genre in enumerate(multilabel_binarizer.classes_):
        movies[genre] = y[:, idx]
    del movies['movie_id']
    del movies['movie_name']
    del movies['genre']
    del movies['plot']
    movies_columns = movies.columns.tolist()
    return movies_columns



def prepare_data():
    trainfilename = 'movies_train.csv'
    evalfilename = 'movies_eval.csv'

    if os.path.exists(trainfilename) and os.path.exists(evalfilename):
        train_df = pd.read_pickle(trainfilename)
        eval_df = pd.read_pickle(evalfilename)

        return train_df, eval_df

    #data preprocessing
    meta = pd.read_csv(os.path.join(
        base_path, "movie.metadata.tsv"), sep='\t', header=None)
    meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
    genres = meta[["movie_id", "movie_name", "genre"]]
    
    plots = pd.read_csv(os.path.join(
        base_path, "plot_summaries.txt"), sep='\t', header=None)
    plots.columns = ["movie_id", "plot"]
    
    genres['movie_id'] = genres['movie_id'].astype(str)
    plots['movie_id'] = plots['movie_id'].astype(str)
    movies = pd.merge(plots, genres, on='movie_id')
    
    genres_lists = []
    for i in movies['genre']:
        genres_lists.append(list(json.loads(i).values()))
    movies['genre'] = genres_lists
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit_transform(movies['genre'])
    # transform target variable
    y = multilabel_binarizer.transform(movies['genre'])
    for idx, genre in enumerate(multilabel_binarizer.classes_):
        movies[genre] = y[:, idx]
    movies.to_csv('movies.csv')
    movies_new = pd.read_csv('movies.csv')
    movies = movies_new
    movies_columns = movies.columns
    
    del movies['Unnamed: 0']
    del movies['movie_name']
    del movies['genre']
    
    df = pd.DataFrame()
    df['id'] = movies['movie_id']
    df['labels'] = list(map(list, zip(*[movies[col] for col in movies
                                        if col != 'movie_id' and col != 'plot' and col != 'text'])))
    df['text'] = movies['plot'].apply(lambda x: x.replace('\n', ' '))
    
    TRAIN_VAL_RATIO = 0.8
    LEN = df.shape[0]
    SIZE_TRAIN = int(TRAIN_VAL_RATIO*LEN)
    
    train_df = df[:SIZE_TRAIN].drop(labels='id', axis=1)
    eval_df = df[SIZE_TRAIN:]

    train_df.to_pickle(trainfilename)
    eval_df.to_pickle(evalfilename)

    return train_df, eval_df


def prepare_train_data_for_coocurences():

    #data preprocessing
    meta = pd.read_csv(os.path.join(
        base_path, "movie.metadata.tsv"), sep='\t', header=None)
    meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
    genres = meta[["movie_id", "movie_name", "genre"]]
    
    plots = pd.read_csv(os.path.join(
        base_path, "plot_summaries.txt"), sep='\t', header=None)
    plots.columns = ["movie_id", "plot"]
    
    genres['movie_id'] = genres['movie_id'].astype(str)
    plots['movie_id'] = plots['movie_id'].astype(str)
    movies = pd.merge(plots, genres, on='movie_id')
    
    genres_lists = []
    for i in movies['genre']:
        genres_lists.append(list(json.loads(i).values()))
    movies['genre'] = genres_lists
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit_transform(movies['genre'])
    # transform target variable
    y = multilabel_binarizer.transform(movies['genre'])
    for idx, genre in enumerate(multilabel_binarizer.classes_):
        movies[genre] = y[:, idx]
        
    #del movies['Unnamed: 0']
    del movies['movie_name']
    del movies['movie_id']
    del movies['plot']
    del movies['genre']
    
    df = movies
    TRAIN_VAL_RATIO = 0.8
    LEN = df.shape[0]
    SIZE_TRAIN = int(TRAIN_VAL_RATIO*LEN)
    
    train_df = df[:SIZE_TRAIN]

    return train_df


def coocurences(t):
    df = prepare_train_data_for_coocurences()
    cooccurrence_matrix = np.dot(df.transpose(),df)
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
    tuples = {}
    result = np.where(cooccurrence_matrix_percentage > t)
    listOfCoordinates= list(zip(result[0], result[1]))
    for cord in listOfCoordinates:
        if cord[0] != cord[1]:
            tuples[(df.columns.tolist()[cord[0]], 
                          df.columns.tolist()[cord[1]])] = cooccurrence_matrix_percentage[cord[0]][cord[1]]

    return tuples

def encode_with_bert(train_df, eval_df, max_seq_len = 50):
    #bert-serving-start -model_dir L-12_H-768_A-12/ -num_worker=2
    filename = f'max_seq_len_{max_seq_len}'
    trainfile = f'{filename}_train.npy'
    testfile = f'{filename}_test.npy'
    if not os.path.exists(trainfile) or not os.path.exists(testfile):
        args = get_args_parser().parse_args(['-model_dir', 'uncased_L-12_H-768_A-12/',
                                             '-max_seq_len', f'{max_seq_len}',
                                             '-port', '5555',
                                             '-fp16', '-xla',
                                             '-num_worker', '1'])
        server = BertServer(args)
        server.start()

        bc = BertClient()
        x_train = bc.encode(list(train_df.text))
        x_test = bc.encode(list(eval_df.text))

        shut_args = get_shutdown_parser().parse_args(
            ['-ip', 'localhost', '-port', '5555', '-timeout', '5000'])
        BertServer.shutdown(shut_args)
    
        np.save(trainfile, x_train)
        np.save(testfile, x_test)
    else:
        x_train = np.load(trainfile)
        x_test = np.load(testfile)

    return x_train, x_test


def encode_with_bert_sample(sample, max_seq_len = 50):
    #bert-serving-start -model_dir L-12_H-768_A-12/ -num_worker=2
    args = get_args_parser().parse_args(['-model_dir', 'uncased_L-12_H-768_A-12/',
                                         '-max_seq_len', f'{max_seq_len}',
                                         '-port', '5555',
                                         '-fp16', '-xla',
                                         '-num_worker', '1'])
    server = BertServer(args)
    server.start()

    bc = BertClient()
    sample_enc = bc.encode(sample)

    shut_args = get_shutdown_parser().parse_args(
        ['-ip', 'localhost', '-port', '5555', '-timeout', '5000'])
    BertServer.shutdown(shut_args)

    return sample_enc



def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


class MultiLabelClassifier:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(768,)))
        # model.add(hub_layer)
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        #zyzus modified 0.48
        #self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(rate=0.8))
        self.model.add(tf.keras.layers.Dense(363, activation='sigmoid'))
        self.model.summary()

    def train(self, x_train, y_train, x_test, y_test, optimizer='adam', loss=[focal_loss(alpha=.10, gamma=2)],
              metrics=[tf.keras.metrics.AUC()], batch_size=128, epochs=100, ):

        if metrics == 'auc':
            metrics = [tf.keras.metrics.AUC()]
        self.model.compile(optimizer, loss, metrics)

        history = self.model.fit(
            x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test))
        results = self.model.evaluate(x_test, y_test, batch_size)
        for name, value in zip(self.model.metrics_names, results):
            print("%s: %.3f" % (name, value))
        return history, results

    def predict(self, x_test):
        return self.model.predict(x_test)

    def test_sample(self, test_data, actual):
        print("actual ground truth={}, predicted={}".format(
            actual, self.model.predict(test_data)))

#@ex.config
#def my_config():
max_seq_len = 256
batch_size = 128
gamma = 2

#@ex.automain
#def train_and_evaluate(max_seq_len, batch_size, gamma):
train_df, eval_df = prepare_data()
y_train = np.array(train_df['labels'].tolist())
y_test = np.array(eval_df['labels'].tolist())

x_train, x_test = encode_with_bert(train_df, eval_df, max_seq_len=max_seq_len)

classifier = MultiLabelClassifier()
history, results = classifier.train(x_train,
                                    y_train,
                                    x_test,
                                    y_test,
                                    optimizer='adam',
                                    loss=[focal_loss(alpha=.10, gamma=gamma)],
                                    metrics='auc',
                                    batch_size=batch_size,
                                    epochs=100
                                    )

predictions = classifier.model.predict(x_test)

t = 0.20
predicted = []
predicted = (predictions >= t).astype(int)
#print(predicted)
f1 = f1_score(y_test,predicted,average='micro')
print("F1 of BERT model is:", f1)

loss, auc = results

#    movies_columns = get_columns_names()
#    pred_df = pd.DataFrame(predicted, columns = movies_columns) 
#    coocs = coocurences(t=0.9)
#    ind = [i[0] for i in coocs]
#    conc = [i[1] for i in coocs]
#    
#    for i, label in enumerate(ind):
#        for index, row in pred_df.iterrows():
#            if row[label] == 1:
#                row[conc[i]] = 1
#    
#                
#    f1_updated = f1_score(y_test,pred_df.to_numpy(),average='micro')
#    print("Updated F1 of BERT model is:", f1_updated) #t = 0.9: 0.495; t = 0.5, 0.8: 0.491

#ex.log_scalar("test_f1", f1)
#ex.log_scalar("train_loss", loss)
#ex.log_scalar("train_auc", auc)

movies_columns = get_columns_names()
pred_df = pd.DataFrame(predicted, columns = movies_columns) 


for index, row in pred_df.iterrows():
    print(eval_df['text'].tolist()[index])
    for label in pred_df.columns:
        if row[label] == 1:
            print(label)
    print('\n')
    