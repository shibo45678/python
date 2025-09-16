#========4.3 评估模型
# 初始化字典
multi_val_performance = {}
multi_test_performance = {}

# 用验证集、测试集评估模型，并返回验证集评估结果（损失值和MAE）evaluate
multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.createValSet,verbose=0)
multi_test_performance['Conv'] = multi_conv_model.evaluate(multi_window.createTestSet, verbose=0)
# 输出评估结果到multi_performance和multi_val_performance字典里
# 字典到key对应不同模型名称，value对应不同模型下的训练结果（指标为损失值-均方误差和MAE）

multi_window.plot(multi_conv_model)
plt.show()