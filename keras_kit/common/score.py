# _*_ coding: utf-8 _*_

from . import bilstm_cnn_crf


if __name__ == '__main__':
    EPOCHS = 10
    model, (train_x, train_y), (test_x, test_y) = bilstm_cnn_crf.create_model()
    # train model
    model.fit(train_x, train_y, batch_size=16, epochs=EPOCHS, validation_data=[test_x, test_y])
    model.save('model/bilstm_cnn_crf.h5')
