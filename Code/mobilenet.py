import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as sk_mae
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dont allocate entire vram initially
set_session(tf.Session(config=config))
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

#Keras image processing
IMG_SIZE = (224, 224) # default size for inception_v3
def flow_from_dataframe(in_df, path_col, y_col, **dflow_args):
    img_data_gen = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'reflect',
                              zoom_range=0.25,
                             preprocessing_function = preprocess_input)
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode = 'sparse',**dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

#Read the data
img_dir = "boneage-training-dataset/"
csv_path = "boneage-training-dataset.csv"
age_df = pd.read_csv(csv_path)
age_df['path'] = age_df['id'].map(lambda x: img_dir+"{}.png".format(x))
age_df['exists'] = age_df['path'].map(os.path.exists)
age_df['gender'] = age_df['male'].map(lambda x: "male" if x else "female")
mu = age_df['boneage'].mean()
sigma = age_df['boneage'].std()
age_df['zscore'] = age_df['boneage'].map(lambda x: (x-mu)/sigma)
age_df.dropna(inplace=True)

#Examine the distribution of age and gender
print("{} images found out of total {} images".format(age_df['exists'].sum(),age_df.shape[0]))
#print(age_df.sample(5))
age_df[['boneage','male','zscore']].hist()
#plt.show()

#Split into training testing and validation datasets
age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
raw_train_df, test_df = train_test_split(age_df, 
                                   test_size = 0.2, 
                                   random_state = 2018,
                                   stratify = age_df['boneage_category'])
raw_train_df, valid_df = train_test_split(raw_train_df, 
                                   test_size = 0.1,
                                   random_state = 2018,
                                   stratify = raw_train_df['boneage_category'])
#Balance the distribution in the training set
train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(500, replace = True)).reset_index(drop=True)
#print(train_df.sample(5))
#train_df[['boneage', 'male']].hist(figsize = (10, 5))
#plt.show()
train_size=train_df.shape[0]
valid_size=valid_df.shape[0]
test_size=test_df.shape[0]
print("# Training images:   {}".format(train_size))
print("# Validation images: {}".format(valid_size))
print("# Testing images:    {}".format(test_size))

#Make image batch generators
train_gen = flow_from_dataframe(train_df, 
                             path_col = 'path',
                            y_col = 'zscore', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 10)
# Used a fixed dataset as validation set for monitoring training progress and saving best model
valid_X, valid_Y = next(flow_from_dataframe(valid_df, 
                             path_col = 'path',
                            y_col = 'zscore', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = valid_size)) # one big batch
IMG_SHAPE = valid_X[0,:,:,:].shape
print("Image shape: "+str(IMG_SHAPE))


#Model definition
base_iv3_model = MobileNet(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')
#base_iv3_model.trainable = False
bone_age_model = Sequential()
bone_age_model.add(base_iv3_model)
bone_age_model.add(GlobalAveragePooling2D())
bone_age_model.add(Dropout(0.5))
bone_age_model.add(Dense(1024, activation = 'tanh'))
bone_age_model.add(Dropout(0.25))
bone_age_model.add(Dense(1, activation = 'linear')) # linear is what 16bit did

#Compile model
def mae_months(in_gt, in_pred):
    return mean_absolute_error(mu+sigma*in_gt, mu+sigma*in_pred)

bone_age_model.compile(optimizer = 'adam', loss = 'mse', metrics = [mae_months])
bone_age_model.summary()

#Model Callbacks
weight_path="bone_age_weights_trainable_mobilenet.best.hdf5" # saved_model_name
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

#Evaluate model on test dataset
if not os.path.exists(weight_path):
    bone_age_model.fit_generator(train_gen, 
                                  steps_per_epoch=train_size/10,
                                  validation_data = (valid_X, valid_Y), 
                                  epochs = 50, 
                                  callbacks = callbacks_list,
                                  verbose=1)
else:
    bone_age_model.load_weights(weight_path)
# Used a fixed dataset as testing set for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(test_df, 
                             path_col = 'path',
                            y_col = 'zscore', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = test_size)) # one big batch
pred_Y = mu+sigma*bone_age_model.predict(test_X,batch_size=25,verbose=1)
test_Y_months = mu+sigma*test_Y
print("Mean absolute error on test data: "+str(sk_mae(test_Y_months,pred_Y)))

fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.plot(test_Y_months, pred_Y, 'r.', label = 'predictions')
ax1.plot(test_Y_months, test_Y_months, 'b-', label = 'actual')
ax1.legend()
ax1.set_xlabel('Actual Age (Months)')
ax1.set_ylabel('Predicted Age (Months)')
#plt.show()

ord_idx = np.argsort(test_Y)
ord_idx = ord_idx[np.linspace(0, len(ord_idx)-1, num=8).astype(int)] # take 8 evenly spaced ones
fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(ord_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    c_ax.set_title('Age: %2.1f\nPredicted Age: %2.1f' % (test_Y_months[idx], pred_Y[idx]))
    c_ax.axis('off')
plt.show()