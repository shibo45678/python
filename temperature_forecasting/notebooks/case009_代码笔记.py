## 9 CNN卷积神经网络+LSTM 长短期记忆网络

# 1 数据预处理
  # 1.3 对数据进行二次取样             ***
  # 1.4 处理时间列数据   sin cos      ***
  # 1.5 处理风向与风速列数据           ***
  # 1.6 划分数据集
  # 1.7 数据标准化
  # 1.8 绘制小提琴图                 ***
# 2 数据窗口类与方法定义
# 3 构建窗口数据
    # 3.1 使用WindowGenerator类构数据窗口
    # 3.2 构建训练集、验证集和测试集
# 4 CNN模型预测
# 5 LSTM模型预测
# 6 数据窗口类与方法详细步骤解析



import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS' # 设置中文显示

# from matplotlib.font_manager import FontProperties
# font_set = FontProperties(fname=r"Macintosh HD/System/Library/Fonts/Supplemental/Arial Unicode.ttf", size = 15)


df = pd.read_csv('/Users/shibo/PycharmProjects/pythonProject/mooc/6 神经网络/第9章 CNN/data_climate.csv')
print(df)
# 查看第一行P列对应的数据类型
type(df.iloc[0][1])
# 将字符型数据彻底改成浮点型数据，便于后续计算  .astype(float)
df[[ 'p', 'T', 'Tpot', 'Tdew','rh', 'VPmax', 'VPact', 'VPdef', 'sh','H2OC', 'rho', 'wv', 'max. wv','wd']] = df[[ 'p', 'T', 'Tpot', 'Tdew','rh', 'VPmax', 'VPact', 'VPdef', 'sh','H2OC', 'rho', 'wv', 'max. wv','wd']].astype(float)


#======= 1.3 对数据进行二次取样 （原数据每隔10分钟收集一次数据，本案例只需要每小时数据）
df = df[5::6] # 切片，从第一小时开始（索引5开始），每隔6个记录一次
print(df["Date Time"])  # 观察该列，数据已经变成每小时的采样




#========1.4 处理时间列数据
# 目标效果：result_climate完成预处理后的文件
result_df = pd.read_csv('/Users/shibo/PycharmProjects/pythonProject/mooc/6 神经网络/第9章 CNN/result_climate.csv')
print(result_df[['Day sin','Day cos','Year sin','Year cos']])

# 处理步骤：
    # a.将'Data Time'列数据从str数据类型转换为日期-时间（datatime)数据类型，保存到新变量data_time中（原df中删除'Data Time'列）
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(date_time)
    # b.将data_time中数据转换为时间戳格式的数据 datetime.datetime.timestamp
datetime.datetime.timestamp(date_time[5])
timestamp_s = date_time.map(datetime.datetime.timestamp)
print(timestamp_s)
    # c.将时刻序列映射为正弦曲线序列
day = 24*60*60         # 一天多少秒
year = (365.2425)*day  # 一年多少秒

df['Day sin'] = np.sin((timestamp_s * 2 * np.pi) / day)
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
print(df[['Day sin','Day cos','Year sin','Year cos']])
    # d.将转换结果可视化
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('时间[单位：时]（Time [h]）')
plt.title('一天中的时间信号（Time of day signal）')
plt.show()

#========1.5 处理风向与风速列数据
# 目标效果：处理前用极坐标（风速m/s）和风向（0-360）来描述风的强度和方向，处理后用正交坐标系的两个维度（x轴和y轴）上风的强度来描述上述风的强度和方向
print(df[['wv','max. wv','wd']]) # 平均风速、最大风速、风向（角度制）
print(result_df[['Wx','Wy','max Wx','max Wy']])

# 处理步骤：
        # a.处理风速列的异常值（在用来描述风速的wv列和max.wv,存在风速-9999的情况，替换成0
print(df[df['wv']<0]['wv']) # 查看平均风速列小于0的异常值（只看df['wv']这一列）
print(df[df['max. wv']<0]['max. wv']) # 查看最大风速列小于0的异常值
            # 将-9999替换为0 （两种结果一致）
df['wv'][df['wv'] == -9999.0]=0 # 错 df [df['wv'] == -9999.0] ['wv']=0

max_wv = df['max. wv']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
            # 上述改动反映到数据框
print(df['wv'].min())
print(df['max. wv'].min())
        # b.将风向和风速列数据转换为风矢量，重新存入原数据框中
            # 2D直方图--通过可视化的方式解释风矢量类型的数据由于原表风速和风向数据的原因

            # 原表风速和风向数据
plt.hist2d(df['wd'], df['wv'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('风向 [单位：度]')
plt.ylabel('风速 [单位：米/秒]')
plt.show()

            # 风矢量类型的数据
wv = df.pop('wv')                     # 先抓出 再丢了 将df中的wv列保存到wv中，并从原来的df中删除
max_wv = df.pop('max. wv')
wd_rad = df.pop('wd')*np.pi / 180     # 原df里面的风向由角度制转换为弧度制，保存到wd_rad，原来的删掉

df['Wx'] = wv*np.cos(wd_rad)          # 计算平均风力的x和y分量，保存到df的'Wx'列和'Wy'列中
df['Wy'] = wv*np.sin(wd_rad)

df['max Wx'] = max_wv*np.cos(wd_rad)  # 计算最大风力的x和y分量，保存到df的'Wx'列和'Wy'列中
df['max Wy'] = max_wv*np.sin(wd_rad)

plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('风的X分量[单位：m/s]')
plt.ylabel('风的Y分量[单位：m/s]')
plt.show()

plt.hist2d(df['max Wx'], df['max Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('最大风的X分量[单位：m/s]')
plt.ylabel('最大风的Y分量[单位：m/s]')
plt.show()
# 对比两图，分解后有利于我们观察风的状况：找到原点（0，0），
# 假设向上为北，那么南方向的风出现次数较多，此外我们还可以观察到东北-西南方向的风

#========1.6 划分数据集 (取数据集中70%的数据作为训练集，取数据集中20%作为验证集，取数据集中10%作为测试集）
n = len(df)  # 数据框长度 print(df.values.shape) (70091, 19)
train_df = df[0:int(n*0.7)]        # 最开始少量数据iloc的变形 / split(train-val)
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

#========1.7 数据标准化 （原值-均值）/ 标准差
# 先划分数据集 再数据标准化
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std # 对所有集中数据进行标准化
print(df_std)
# 数据标准化（另外的"移动平均值"？？）
# 使用训练集的数据来计算均值和标准差，原因是：因为在训练模型时，不能访问训练集未来时间点的值，
# 而且这种标准化的方法也应该使用移动平均值的方法来完成。但在这里非重点，且验证集和测试集能确保获得（某种程度上）真实的指标

#========1.8 绘制小提琴图
            # 将上述结果变成两列，列的信息提取到行
df_std = df_std.melt(var_name='Column', value_name='Normalized')
print(df_std)

plt.figure(figsize=(12, 6))   # seaborn
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.show()
# 小提琴图：每个小提琴展示了原数据中每一列数据的统计特征，例如第二个小提琴表示列温度列数据可能出现的取值，以及这些取值出现的概率。
        #  上下端点纵坐标值是可能的取值，每个取值在横坐标上的宽度表示该取值出现的概率。
        #  例如第二个小提琴中，越宽的地方表示温度出现概率越高。
        #  每个小提琴里的矩形上下端点表示四分之一和四分之三位数的位置，白点表示二分位数位置




##======================================================= 2 数据窗口类与方法定义=======================================================
class WindowGenerator():  # 创建窗口
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}

        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_indices = np.arange(self.total_window_size)[0:input_width]

        self.label_start = self.total_window_size - self.label_width
        self.label_indices = np.arange(self.total_window_size)[self.label_start:]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

 # 划分数据窗口
 # 只需让刚刚创建好的single_window直接调用createTrainSet()方法或createValSet()方法或createTestSet（）方法即可
def split_window(self, features):
    inputs = features[:, 0:self.input_width, :]
    labels = features[:, self.label_start:, :]

    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

WindowGenerator.split_window = split_window


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)

    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32, )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def createTrainSet(self):
    return self.make_dataset(self.train_df)


@property
def createValSet(self):
    return self.make_dataset(self.val_df)


@property
def createTestSet(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    result = getattr(self, '_example', None)
    if result is None:
        result = next(iter(self.createTrainSet))
        self._example = result
    return result


WindowGenerator.createTrainSet = createTrainSet
WindowGenerator.createValSet = createValSet
WindowGenerator.createTestSet = createTestSet
WindowGenerator.example = example


def plot(self, model=None, plot_col='T', max_subplots=3):
    inputs, labels = self.example
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n + 1)
        plt.ylabel(f'标准化后的 {plot_col}')

        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='green', s=64)

        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='orange', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')


WindowGenerator.plot = plot




##=============================================================3 构建窗口函数=============================================================

#=======3.1 使用WindowGenerator类构建数据窗口
single_window = WindowGenerator(input_width=6, label_width=5, shift=1,label_columns=['T'])
# 历史数据6个数，倒着取5个 shift?
print(single_window) # 总窗口大小7

multi_window = WindowGenerator(input_width=6, label_width=5, shift=24)
print(multi_window)

#========3.2 构建训练集、验证集和测试集
print('训练数据:')
print(multi_window.createTrainSet)
print('验证数据：')
print(multi_window.createValSet)
print('测试数据：')
print(multi_window.createTestSet)

for train_inputs, train_labels in multi_window.createTrainSet.take(1):
    print(f'Inputs shape (batch, time, features): {train_inputs.shape}')
    print(f'Labels shape (batch, time, features): {train_labels.shape}')




##=============================================================4 CNN模型预测=============================================================
# =====4.1 构建模型
# 基于历史六个时间点的天气情况（6行19列）预测经过24小时（shift=24)未来5个时间点的天气状况
multi_conv_model = tf.keras.Sequential([
    # 卷基层：输入数据是三维的tensor形式（conv1d是单通道，适合处理文本序列，conv2d适合处理图像）
    # 【32，6，19】==>【32,4,64】
    # filters 过滤器个数为64（通道）
    # kernel_size 只需要指定一维，长度为3，剩下的与窗口宽度一致19，卷积核的大小3*19。
    # strides 每次卷积核移动一步（6*19会卷出来4个数，一共64个）
    # 卷积核只能从上往下走，不能从左往右走，即只能按照时间点的顺序
    # param个数 3712（3*19*64=64）= 卷积核长*宽*通道数+每个通道一个偏置（偏置：控制每个神经元被激活的容易程度）
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'),
    # flatten层：将输入拉平，即把多维的输入一维化，用在从卷基层到全联接层的过渡
    # 【32，4，64】==> 【32,256】 一次4个，一共64个通道
    tf.keras.layers.Flatten(),
    # 【32，256】==>【32，95】  5*19
    # 权重初始化的方法：全零初始化
    tf.keras.layers.Dense(5 * 19, kernel_initializer=tf.initializers.zeros),
    # reshape层，将输入shape转换为特定shape
    # 【32,95】==>【32,5,19】
    # param数 24415 =4*64*95+95 = 256*95+95  全联接+偏置
    tf.keras.layers.Reshape([5, 19])
])



#========4.2 训练模型
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




##=============================================================5 LSTM模型预测=============================================================
#======5.1 构建模型
multi_lstm_model = tf.keras.Sequential([
    # 64代表LSTM输出结果的维度：设置激活函数tanh;
    # return_sequences=false 设置只在最后一个时间步产生输出
    # shape[32,6,19]==>[32,64]
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=False),
    # dense  shape[32,5*19]
    # kernel_initializer
    tf.keras.layers.Dense(5 * 19, kernel_initializer=tf.initializers.zeros),
    # 输出层 shape [32,5,19]
    tf.keras.layers.Reshape([5, 19])
])
    # tanh函数 （将一个实数映射到（-1 1）的区间
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
inputs=np.arange(-10,10,0.1)
outputs=tanh(inputs)
plt.plot(inputs,outputs)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#======5.2 训练模型
history = compile_and_fit(multi_lstm_model, multi_window)
# LSTM 层的参数总数【（64+19）*64 + 64】*4 == 【（上一轮输出+输入）*（全联接输出）+（输出层偏置）】*4层（遗忘门*1+记忆门*2+输出门*1）
# dense 1  参数总数 64*95+95=6175 全联接
multi_lstm_model.summary()


# =====5.3 评估模型 （编译省略 之前做过）CNN有个df compile
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.createValSet,verbose=0)
multi_test_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.createTestSet, verbose=0)

multi_window.plot(multi_lstm_model)
plt.show()



# 如果return_sequences=True  (LSTM2)
multi_lstm_model2 = tf.keras.models.Sequential([
    # 输出全部序列，包含6个时间步输出的序列，即每一个cell的h_t
    # 窗口那边 5个输出只要1个输出，操作，加一层LSTM
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),  # 5个输出
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=False), # 只要最后1个输出 # 总参数计算注意（自己的输入+上一层输入的变化）
    tf.keras.layers.Dense(5 * 19, kernel_initializer=tf.initializers.zeros), # shape 32*95
    tf.keras.layers.Reshape([5, 19]) # 输出层 shape 32*5*19
])

history = compile_and_fit(multi_lstm_model2, multi_window)
multi_lstm_model2.summary()

multi_val_performance['LSTM2'] = multi_lstm_model2.evaluate(multi_window.createValSet,verbose=0)
multi_test_performance['LSTM2'] = multi_lstm_model2.evaluate(multi_window.createTestSet,verbose=0)



# =====比较CNN和LSTM的预测效果
print(multi_val_performance) # 展示验证集的评估效果（conv/LSTM1/LSTM2)
print(multi_test_performance) # 展示测试集的评估效果
# 找出测试值MAE所属的索引
metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
print(metric_index)
multi_val_performance.values()
# 根据MAE的索引遍历验证集的评估结果，返回所有模型的MAE测量值 *
val_mae=[v[metric_index] for v in multi_val_performance.values()]
print(val_mae)
test_mae = [v[metric_index] for v in multi_test_performance.values()]
print(test_mae)

x = np.arange(len(multi_test_performance))  # 3个模型
plt.ylabel('mean_absolute_error')  # 指定纵轴标签
# 画两条竖状条形，x指定柱体在X轴上的坐标位置，height指定柱体的高度（相当y)，width指定柱体宽度 与bar和barh的width和height的区别
plt.bar(x=x-0.17, height=val_mae, width=0.3, label='Validation')   # bar 柱状图
plt.bar(x=x+0.17, height=test_mae, width=0.3, label='Test')
plt.xticks(ticks=x, labels=multi_test_performance.keys(),rotation=45)  # 指定X轴的刻度 用performance的key对应横坐标3个标签
_ = plt.legend()

##==================
# tf.keras.preprocessing.timeseries_dataset_from_array基于时间序列创建滑动窗口的dataset并以数组形式输出
ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=total_window_size, # 设置输出序列的长度
      sequence_stride=1, # 连续输出序列之间的周期
      shuffle=True,
      batch_size=32,)
ds

# 打印调用tf.keras.preprocessing.timeseries_dataset_from array 方法后的维度结果
ds_list = list(ds.as_numpy_iterator())
print(len(ds_list),len(ds_list[0]),len(ds_list[0][0]),len(ds_list[0][0][0]))
# 7*19 整个窗口高（6*19。5*1）

# 1534 32 7 19
#一共1534组batch（训练样本） ，一组batch有32个训练样本的值，一个训练样本的值对应7行数据。一行数据是一个时刻的天气情况

tf_lst = tf.keras.preprocessing.timeseries_dataset_from_array(
         lst,targets=None,sequence_length = 10,# 设置输出序列的长度
         sequence_stride=3, # 连续输出序列之间的周期
         sampling_rate=2,# 设置序列内连续时间步之间的间隔
         batch_size=3, # 设置每批中时间序列样本的数量
         )

temp_lst = list(tf_lst.as_numpy_iterator())
temp_lst

# [array([[ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18],
#         [ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21],
#         [ 6,  8, 10, 12, 14, 16, 18, 20, 22, 24]]),
#  array([[ 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
#         [12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
#         [15, 17, 19, 21, 23, 25, 27, 29, 31, 33]]),
#  array([[18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
#         [21, 23, 25, 27, 29, 31, 33, 35, 37, 39],
#         [24, 26, 28, 30, 32, 34, 36, 38, 40, 42]]),
#  array([[27, 29, 31, 33, 35, 37, 39, 41, 43, 45],
#         [30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
#         [33, 35, 37, 39, 41, 43, 45, 47, 49, 51]]),
#  array([[36, 38, 40, 42, 44, 46, 48, 50, 52, 54],
#         [39, 41, 43, 45, 47, 49, 51, 53, 55, 57],
#         [42, 44, 46, 48, 50, 52, 54, 56, 58, 60]]),
#  array([[45, 47, 49, 51, 53, 55, 57, 59, 61, 63],
#         [48, 50, 52, 54, 56, 58, 60, 62, 64, 66],
#         [51, 53, 55, 57, 59, 61, 63, 65, 67, 69]]),
#  array([[54, 56, 58, 60, 62, 64, 66, 68, 70, 72],
#         [57, 59, 61, 63, 65, 67, 69, 71, 73, 75],
#         [60, 62, 64, 66, 68, 70, 72, 74, 76, 78]]),
#  array([[63, 65, 67, 69, 71, 73, 75, 77, 79, 81],
#         [66, 68, 70, 72, 74, 76, 78, 80, 82, 84],
#         [69, 71, 73, 75, 77, 79, 81, 83, 85, 87]]),
#  array([[72, 74, 76, 78, 80, 82, 84, 86, 88, 90],
#         [75, 77, 79, 81, 83, 85, 87, 89, 91, 93],
        # [78, 80, 82, 84, 86, 88, 90, 92, 94, 96]])]