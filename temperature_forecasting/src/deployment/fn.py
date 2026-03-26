
# # 导出SavedModel + 自定义签名
# import tensorflow as tf
# import os
#
#
# def save_savedmodel(model):
#     input_specs = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype, name=inp.name) for inp in model.inputs]
#     # [<KerasTensor shape=(None, 6, 33), dtype=float32, sparse=False, ragged=False, name=numeric_input>,
#     #  <KerasTensor shape=(None, 6), dtype=float32, sparse=False, ragged=False, name=categorical_segments_input>]
#     # 从模型输入获取每个输入的 TensorSpec
#     input_signature = [tuple(input_specs)]
#
#     # 预热 帮助资源追踪
#     dummy_inputs = [tf.zeros([1] + spec.shape[1:], dtype=spec.dtype) for spec in input_specs]
#     _ = model(dummy_inputs, training=False)
#
#     @tf.function
#     def serving_fn(inputs):
#         # inputs 接收预测窗口形成的元组格式的输入
#         if len(inputs) == 1:
#             numeric = inputs[0]
#             return model([numeric])
#         else:
#             numeric = inputs[0]
#             categoricals = inputs[1:]
#             return model([numeric, *categoricals], training=False)
#
#     concrete_fn = serving_fn.get_concrete_function(input_signature)
#     saved_path = '/Users/shibo/AL/NeuralNetwork/deployment_package/saved_model/'
#
#     os.makedirs(saved_path, exist_ok=True)
#     tf.saved_model.save(model, saved_path, signatures={'serving_default': concrete_fn})
#
#     return
#
# path = '/Users/shibo/AL/NeuralNetwork/saved_model/single_lstm1_20260325_165112/tf_checkpoints_stage0/epoch_2/model_stage0.keras'
# model = tf.keras.models.load_model(path)
#
#
# save_savedmodel(model)