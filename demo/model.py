from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from true_test import create_train_dev_set

import time
import os


class Ls_model:

    def __init__(self, embedding_dim, max_sequence_length, numbber_lstm_units, numer_dense, rate_drop_lstm,
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = numbber_lstm_units
        self.rate_drop_lstm = rate_drop_lstm
        self.numer_dense_units = numer_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio

    def train_model(self, sentence_pair, is_similar, embedding_meta_data, model_save_directory='./'):
        '''

        :param sentence_pair: two sentences
        :param is_similar: label
        :param embedding_meta_data:y
        :param mdoel_save_directory: model save paths
        :return: best_model_path
        '''
        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentence_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)

        if train_data_x1 is None:
            print('no training data')
            return None
        nb_words = len(tokenizer.word_index) + 1

        # creating word embedding layer
        embedding_layer = Embedding(nb_words, self.embedding_dim, weight=[embedding_matrix],
                                    input_length=self.max_sequence_length, trainable=False)

        # creating LSTM Encoder
        lstm_layer = Bidirectional(
            LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm))

        # creating LSTM Encoder layer for First sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_sequence_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedding_sequence_1)

        # creating LSTM Encoder for second sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedding_sequence_2 = embedding_layer(sequence_2_input)
        x2 = lstm_layer(embedding_sequence_2)

        # merge two LSTM encoder vector from sentence to networmk
        # pass dense layer applying dropout and batch normalisation
        merged = concatenate([x1, x2])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
        model.compile(loss='binary-crossentropy', optimizer='nadam', metrics=['acc'])

        early_stop = EarlyStopping(monitor='val_loss', patienc=3)

        STAMP = 'LSTM_%d_%d_%.2f_%.2f' % (
        self.number_lstm_units, self.numer_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time()))

        if os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + 'logs/{}'.format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels)
                  , epochs=200, batch_size=64, shuffle=True, callbacks=[early_stop, model_checkpoint, tensorboard])

        return bst_model_path

    def update_model(self, save_model_path, new_sentences_pair, is_similiar, embedding_meta_data):
        tokenizer = embedding_meta_data['tokenizer']

        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                               is_similiar, self.max_sequence_length,
                                                                               self.validation_split_ratio)

        model = load_model(save_model_path)
        model_file_name = save_model_path.split('/')[-1]
        new_model_checkpoint_path = save_model_path.split('/')[:2] + str(int(time.time())) + '/'

        new_model_path = new_model_checkpoint_path + model_file_name

        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + 'log/{}'.format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=3, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path
