def TrainingCnn(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1):
        """训练模型"""
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        return history

    # ========4.2 训练模型


MAX_EPOCHS = 20  # 设置训练的总轮数


def compile_and_fit(model, window):
    # EarlyStopping作用：当监测指标停止改善时，停止训练
    # monitor监测指标，patience 没有进步的训练轮数，在这之后训练停止，mode=min 当监测指标停止减少时训练停止（维持最小值）
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

    # 编译模型 设置损失函数：均方误差函数（MSE），优化器Adam（随机梯度下降），metrics设置模型检验方法 ：平均绝对值误差 MAE
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    # 训练模型 （设置输入：训练数据集， 设置训练总轮数  设置验证数据集  设置提前结束训练
    # verbose 设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录 2 epoch每轮输出一行记录
    history = model.fit(window.createTrainSet, epochs=MAX_EPOCHS, validation_data=window.createValSet,
                        callbacks=[early_stopping], verbose=2)
    return history


history = compile_and_fit(multi_conv_model, multi_window)
multi_conv_model.summary()  # 出来一个表 显示每一层参数个数


