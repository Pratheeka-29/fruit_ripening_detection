
```markdown
Image-Based Detection of Chemically Induced Ripening in Fruits
===============================================================

This project presents a non-invasive deep learning-based approach to detect artificially ripened fruits (specifically bananas) using image classification and hybrid feature extraction techniques. It addresses a critical public health issue — the use of harmful chemicals such as calcium carbide in fruit ripening.

-------------------------------------------------------------------------------

Abstract
--------

The widespread use of chemicals such as calcium carbide for artificial ripening of fruits poses a serious health hazard. Traditional detection methods are either manual or lab-based, making them unsuitable for large-scale market screening. This project introduces an image-based detection system that uses computer vision and deep learning to automatically differentiate between naturally and chemically ripened bananas.

Using pre-trained models (InceptionV3 and EfficientNetB0) for hybrid feature extraction and a stacked ensemble of machine learning classifiers for final prediction, the model achieves accurate classification without chemical testing. Results demonstrate strong reliability in identifying subtle visual differences caused by chemical ripening.

-------------------------------------------------------------------------------

Technologies Used
-----------------

- Python 3.10+
- TensorFlow and Keras
  - InceptionV3
  - EfficientNetB0
- OpenCV (image loading and preprocessing)
- scikit-learn (classifiers, stacking, evaluation)
- XGBoost (gradient boosting classifier)
- Matplotlib (visualization)
- TQDM (progress display)

-------------------------------------------------------------------------------

Model Architecture
------------------

1. **Feature Extraction**
   - InceptionV3 and EfficientNetB0 (pre-trained on ImageNet) are used to extract deep features from 299x299 banana images.
   - Their output vectors are concatenated to form a hybrid feature representation.

2. **Classification**
   - A Stacking Ensemble Classifier is trained using the following base learners:
     - XGBoost
     - K-Nearest Neighbors
     - Logistic Regression
     - Gradient Boosting
   - The final estimator is XGBoost.

-------------------------------------------------------------------------------

How the System Works
---------------------

1. Load images from the dataset folders: `with_carbide/` and `without_carbide/`
2. Preprocess images using InceptionV3 and EfficientNetB0 preprocessing pipelines.
3. Extract deep features from each image.
4. Concatenate feature vectors to form hybrid representations.
5. Train a stacking classifier on the feature vectors and labels.
6. Predict new input images using the trained model and display results with confidence scores.

-------------------------------------------------------------------------------

Model Evaluation and Results
----------------------------

- Overall Model Accuracy: 93.03%
- Classification Report:

```

Precision, Recall, F1-score for both classes: 'with carbide' and 'without carbide'

```

Sample Outputs:

Input Image: result_1.jpg  
Prediction: Without Carbide  
Confidence: 94.85%

Input Image: result_2.jpg  
Prediction: With Carbide  
Confidence: 91.42%

-------------------------------------------------------------------------------

Installation and Running Instructions
-------------------------------------

1. Clone the repository:
```

git clone [(https://github.com/Pratheeka-29/fruit_ripening_detection.git)](https://github.com/Pratheeka-29/fruit_ripening_detection.git)
cd FruitRipeningDetection

```

2. Install required libraries:
```

pip install -r requirements.txt

````

3. Run the project:
- Option 1: Use the Jupyter notebook `main.ipynb` to train and evaluate the model.
- Option 2: Use terminal commands to run scripts:
  ```
  python model_training.py
  python prediction.py --image path_to_image.jpg
  ```

-------------------------------------------------------------------------------

References
----------

- Pretrained models from `tensorflow.keras.applications`

-------------------------------------------------------------------------------

Disclaimer
----------

This tool is developed solely for academic and research purposes. It is not certified for official food safety testing and should not be used as a substitute for chemical laboratory analysis.

-------------------------------------------------------------------------------

Author
------

Name: Pratheeka K G  
Role: Machine Learning Intern (2025)  
Email: pratheekakg@example.com  
GitHub: https://github.com/Pratheeka-29
````

---


