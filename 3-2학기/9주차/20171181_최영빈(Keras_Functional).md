### Functional API를 이용한 NN 구축

```python
# mlp for binary classification
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import Input

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
df = read_csv(path, header=None)

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]

# ensure all data are floating point values
X = X.astype('float32')

# encode strings to integer
y = LabelEncoder().fit_transform(y)

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# determine the number of input features
n_features = X_train.shape[1]

# define functional model #
x_in = Input(shape=(n_features,))
x_hl1 = Dense(10, activation='relu', kernel_initializer='he_normal')(x_in)
x_hl2 = Dense(8, activation='relu', kernel_initializer='he_normal')(x_hl1)
x_out = Dense(1, activation='sigmoid')(x_hl2)

functional_model = Model(inputs=x_in, outputs=x_out)

# compile the functional model
functional_model.compile(optimizer='adam', loss='binary_crossentropy',
                         metrics=['accuracy'])

# fit the functional model
functional_model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# evaluate the model
func_loss, func_acc = functional_model.evaluate(X_test, y_test, verbose=0)
print('Functional Accuracy: %.3f' % func_acc)

# make a prediction
row = [1, 0, 0.99539, -0.05889, 0.85243, 0.02306, 0.83398, -0.37708, 1, 0.03760, 0.85243, -0.17755, 0.59755, -0.44945, 0.60536, -0.38223, 0.84356, -0.38542,
       0.58212, -0.32192, 0.56971, -0.29674, 0.36946, -0.47357, 0.56811, -0.51171, 0.41078, -0.46168, 0.21266, -0.34090, 0.42267, -0.54487, 0.18641, -0.45300]
yfunchat = functional_model.predict([row])
print('Predicted: %.3f' % yfunchat)

```

    (235, 34) (116, 34) (235,) (116,)


    2021-10-20 16:03:44.371538: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.


    Functional Accuracy: 0.905
    Predicted: 0.988


    2021-10-20 16:03:49.400629: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
    2021-10-20 16:03:49.462664: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
