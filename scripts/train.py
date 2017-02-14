import argparse
import os

import h5py

from keras.callbacks import (Callback, CSVLogger, EarlyStopping,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop


class ModelReset(Callback):
    def on_epoch_end(self, epoch, logs, **kwargs):
        print('reseting model states, end-epoch: ', epoch)
        self.model.reset_states()


def train(experiment_id, input_dataset, num_cells, num_layers, dropout_probability, batch_size, timesteps, epochs, lr, loss_weight, snapshot_freq, lrp_gain, lrp_patience, es_patience, feature_size=4096, hdf5_ds_name='c3d_features'):
    print('Experiment ID {}'.format(experiment_id))

    print('number of cells: {}'.format(num_cells))
    print('number of layers: {}'.format(num_layers))
    print('dropout probability: {}'.format(dropout_probability))

    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}'.format(lr))
    print('loss weight for background class: {}'.format(loss_weight))

    store_weights_root = 'data/model_snapshot'
    store_weights_file = 'lstm_activity_classification_' + str(experiment_id) + '_e{epoch:03d}.hdf5'
    logging_file = os.path.join(store_weights_root, experiment_id + '.tsv')
    print('lr-plateau gain: {}'.format(lrp_gain))
    print('lr-plateau patience: {}'.format(lrp_patience))
    print('early-stopping patience: {}'.format(es_patience))
    print('logging file: {}\n'.format(logging_file))

    weight_format = os.path.join(store_weights_root, store_weights_file)
    callbacks = [ModelCheckpoint(weight_format, monitor='val_loss',
                                 save_weights_only=True, verbose=1,
                                 period=snapshot_freq, save_best_only=True),
                 ModelReset()]
    callbacks += [CSVLogger(logging_file, separator='\t')]
    if lrp_gain > 0 or lrp_patience > 0:
        callbacks += [ReduceLROnPlateau(monitor='val_loss', factor=lrp_gain,
                                        patience=lrp_patience, verbose=1,
                                        mode='auto')]
    if es_patience > 0:
        callbacks += [EarlyStopping(monitor='val_loss', patience=es_patience,
                                    verbose=1)]

    print('Compiling model')
    input_features = Input(batch_shape=(batch_size, timesteps, feature_size,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=dropout_probability)(input_normalized)
    lstms_inputs = [input_dropout]
    for i in range(num_layers):
        previous_layer = lstms_inputs[-1]
        lstm = LSTM(num_cells, return_sequences=True, stateful=True, name='lsmt{}'.format(i+1))(previous_layer)
        lstms_inputs.append(lstm)

    output_dropout = Dropout(p=dropout_probability)(lstms_inputs[-1])
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'],
        sample_weight_mode='temporal')
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File(input_dataset, 'r')
    X = f_dataset['training']['vid_features']
    Y = f_dataset['training']['output']
    print('Loading Sample Weights...')
    sample_weight = f_dataset['training']['sample_weight'][...]
    sample_weight[sample_weight != 1] = loss_weight
    print('Loading Validation Data...')
    X_val = f_dataset['validation']['vid_features']
    Y_val = f_dataset['validation']['output']
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output shape: {}'.format(Y_val.shape))
    print('Sample Weights shape: {}'.format(sample_weight.shape))

    model.fit(X,
              Y,
              batch_size=batch_size,
              validation_data=(X_val, Y_val),
              sample_weight=sample_weight,
              verbose=1,
              nb_epoch=epochs,
              shuffle=False,
              callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the RNN ')

    parser.add_argument('--id', dest='experiment_id', default=0, help='Experiment ID to track and not overwrite resulting models')

    parser.add_argument('-i', '--input-data', type=str, dest='input_dataset', default='data/dataset/dataset_stateful.hdf5', help='File where the stateful dataset is stored (default: %(default)s)')

    parser.add_argument('-n', '--num-cells', type=int, dest='num_cells', default=512, help='Number of cells for each LSTM layer (default: %(default)s)')
    parser.add_argument('--num-layers', type=int, dest='num_layers', default=1, help='Number of LSTM layers of the network to train (default: %(default)s)')
    parser.add_argument('-p', '--drop-prob', type=float, dest='dropout_probability', default=.5, help='Dropout Probability (default: %(default)s)')

    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=256, help='batch size used to create the stateful dataset (default: %(default)s)')
    parser.add_argument('-t', '--timesteps', type=int, dest='timesteps', default=20, help='timesteps used to create the stateful dataset (default: %(default)s)')
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=100, help='number of epochs to last the training (default: %(default)s)')
    parser.add_argument('-l', '--learning-rate', type=float, dest='lr', default=1e-5, help='learning rate for training (default: %(default)s)')
    parser.add_argument('-w', '--loss-weight', type=float, dest='loss_weight', default=.3, help='value to weight the loss to the background samples (default: %(default)s)')
    parser.add_argument('-fsz', '--feature-size', type=int, default=4096, help='Input dimension')
    parser.add_argument('-sfq', '--snapshot-freq', type=int, default=5, help='Control snapshot frequency')
    parser.add_argument('-glrp', '--gain-lr-plateau', type=float, dest='lrp_gain', default=0.1, help='Gain for learning rate on plateau')
    parser.add_argument('-plrp', '--patience-lr-plateau', type=int, dest='lrp_patience', default=0, help='Patience for learning rate on plateau')
    parser.add_argument('-pes', '--patience-early-stop', type=int, dest='es_patience', default=0, help='Patience for early stopping')
    parser.add_argument('-hdn', '--hdf5-ds-name', type=str, default='c3d_features', help='Name of HDF5 dataset with C3D features')

    args = parser.parse_args()

    train(**vars(args))
