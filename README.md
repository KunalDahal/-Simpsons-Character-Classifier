
# 🧠 Simpsons Character Classifier

This project builds a convolutional neural network (CNN) to classify characters from **The Simpsons** using TensorFlow, OpenCV, and the `caer` and `canaro` libraries.

## 📁 Dataset

The model is trained on the [Simpsons Characters Dataset](https://www.kaggle.com/datasets/kostastokis/simpsons-dataset), containing over 20,000 images of various characters.

## 🚀 Features

- Loads and preprocesses image data using `caer`
- Uses only the top 10 characters with the most images
- CNN architecture with three convolutional blocks
- Real-time data augmentation
- Validation split to monitor overfitting
- Custom learning rate scheduler
- Prediction on a sample test image

## 🛠️ Requirements

```bash
pip install caer canaro opencv-python tensorflow matplotlib
```

## 🧠 Model Architecture

```text
Input (80x80x1 grayscale) 
→ Conv2D (x2) → MaxPooling → Dropout 
→ Conv2D (x2) → MaxPooling → Dropout 
→ Conv2D (x2) → MaxPooling → Dropout 
→ Flatten → Dropout → Dense(1024) 
→ Output (Dense with Softmax)
```

## 🏋️‍♂️ Training

```python
model.fit(train_gen,
          steps_per_epoch=len(x_train)//BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(x_val, y_val),
          validation_steps=len(y_val)//BATCH_SIZE,
          callbacks=callbacks_list)
```

## 🧪 Testing

To predict the class of a test image:
```python
img = cv.imread("path_to_test_image.jpg")
predictions = model.predict(prepare(img))
print(characters[np.argmax(predictions[0])])
```

## 📊 Results

The model achieves high accuracy on distinguishing between the top 10 characters. Improvements can be made by training longer, tuning hyperparameters, or expanding the number of classes.
