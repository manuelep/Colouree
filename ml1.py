import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
df = pd.read_csv("AirQualityUCI.csv",parse_dates=[['Date', 'Time']])
df=df.drop(['Date_Time'], axis=1)

validation_size = 0.11
seed=7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'


lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
Y_validation = lb.transform(Y_validation)


# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(256, input_dim=3, init='uniform', activation='sigmoid'))
#model.add(Dense(512, input_dim=3, init='normal', activation="sigmoid"))
#model.add(Dense(10, activation="sigmoid"))
model.add(Dense(64, activation="sigmoid"))
#model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 10
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
H = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation),epochs=EPOCHS, batch_size=32)
#H=model.fit(X_train, Y_train, epochs=15, batch_size=10,  verbose=3)
#predictions = model.predict(X_validation[0])
scores = model.evaluate(X_validation, Y_validation)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(X_validation, batch_size=32)
#print(classification_report(Y_validation.argmax(axis=1),
#	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("plot.png")
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)