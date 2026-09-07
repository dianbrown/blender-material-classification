# **ITRI626 practical project** 

### **Deep learning for image classification** 

|**Module**|ITRI626|
|---|---|
|**Lecturer**|Prof. Absalom Ezugwu|
|**Facilitators**|Dr. Sylvester Egwuche and Mr. Bernard Swanepoel|



## **1 Project overview** 

This group project gives students practical experience in applying machine learning and deep neural networks to an image classification problem. Each group must select a suitable topic, prepare and document an image dataset, train and compare classification models, evaluate the results, and communicate the work in a written project report and an oral presentation. 

The complete project contributes 30 marks. The project report contributes 20 marks, and the oral presentation contributes 10 marks. 

## **2 Project scope** 

Each group must complete an image classification project. Each input image must be assigned to one class or, where justified, to more than one class. The group must clearly define the classification task, the target classes, and the intended practical or scientific value of the solution. 

The project must include data selection and preparation, model training, evaluation, comparison, interpretation, and reporting. 

## **3 Group size and responsibilities** 

Students must work in groups of three unless the lecturer approves another arrangement. All group members are responsible for the complete project and must understand the dataset, models, experiments, and results. 

Work may be divided among group members, but the final submission must be coherent. All members must participate in the oral presentation and must be prepared to answer questions about any part of the project. 

## **4 Topic selection** 

Groups may propose any suitable image classification topic. The topic must be approved before substantial implementation begins. 

A suitable topic must have clearly defined classes, enough usable data, a meaningful application, and a scope that can be completed within the semester. 

1 

|**Example domain**|**Possible classifcation directions**|
|---|---|
|Physics and astronomy|Sunspot classifcation, solar radio burst classifcation, galaxy<br>morphology classifcation, or stellar spectrum image classi-<br>fcation.|
|Medical and biomedical<br>imaging|X-ray classifcation, skin lesion classifcation, or microscopy<br>image classifcation. Only de-identifed and ethically usable<br>data may be used.|
|Engineering and manu-<br>facturing|Normal and defective component classifcation, surface de-<br>fect classifcation, or material classifcation.|
|Remote<br>sensing<br>and<br>Earth observation|Land cover classifcation, cloud classifcation, or wildfre and<br>smoke image classifcation.|
|Agriculture and the en-<br>vironment|Plant disease classifcation, fruit ripeness classifcation, or<br>plant and animal species classifcation.|
|General applications|Waste classifcation, food classifcation, document image<br>classifcation, product classifcation, or another approved<br>problem.|



## **5 Dataset requirements** 

A public dataset or public database may be used. Creating a completely new dataset is encouraged when practical, but it is not compulsory. Regardless of the source, the group must demonstrate meaningful understanding and preparation of the data. 

The report and notebook must document the following: 

- The dataset source, access link, licence, and usage conditions. 

- The class definitions and the reasons for selecting those classes. 

- The number of images in each class before and after cleaning. 

- Any removed duplicates, corrupted files, irrelevant samples, or uncertain labels. 

- How the labels were obtained, checked, and corrected. 

- Any class imbalance and how it was handled. 

- Representative examples from every class. 

For data involving people, health information, or sensitive records, the group must use deidentified data and follow all ethical, privacy, and licensing requirements. Data collected from the internet must be used lawfully, and the source must be acknowledged properly. 

## **6 Model requirements** 

Each group must implement and compare at least three image classification models from three different model families. At least one of the three models must use transfer learning with pretrained weights. 

2 

The three main models must represent meaningfully different architectures. Different variants from the same model family do not count as separate main models. For example, ResNet18 and ResNet50 count as one model family, while ViT-B/16 and ViT-L/16 count as another model family. 

Suitable model choices include the following: 

- A small custom convolutional neural network as a baseline. 

- ResNet, DenseNet, or MobileNetV3. 

- EfficientNetV2 or ConvNeXt. 

- Vision Transformer or DeiT. 

- Swin Transformer or another approved image classification architecture. 

All three models must be trained and evaluated using the same data split. The comparison must be fair and clearly explained. Hyperparameters do not have to be identical when the architectures require different settings, but the group must justify the choices and avoid giving one model an unreasonable advantage. 

## **7 Experimental design** 

### **7.1 Training, validation and test data** 

The dataset must be divided into training, validation, and test sets. A split such as 70/15/15 or 80/10/10 may be used, although another split may be justified. The same test set must be used for all three models. 

The split must prevent data leakage. Near-duplicate images or related samples from the same person, patient, object, video, physical sample, observation sequence, or source group must not be divided across the training and test sets. Where such grouping exists, the data must be split according to group identity rather than through random image selection. 

The report must state the random seed, the exact number of images in each split, the class distribution in each split, and the method used to prevent data leakage. 

### **7.2 Preprocessing and augmentation** 

Students must describe all preprocessing steps, such as resizing, normalisation, colour conversion, cropping, or contrast adjustment. 

Any augmentation, such as flips, rotations, random crops, blur, noise, MixUp, or CutMix, must be suitable for the selected domain and must be applied only to the training data. The report must explain why the selected transformations are realistic for the chosen problem. 

### **7.3 Training information and reproducibility** 

The notebook and report must provide enough information for another student to understand and reproduce the experiments. At minimum, the following information must be included: 

3 

- The model name, model variant, and pretrained weights. 

- The input image size. 

- The batch size and any gradient accumulation. 

- The number of epochs and the early stopping settings. 

- The optimiser, learning rate, and learning rate schedule. 

- The weight decay and loss function. 

- The class weights or sampling method, where applicable. 

- The random seed. 

- The software framework and relevant library versions. 

- The hardware used, including the GPU model and available memory, where applicable. 

## **8 Evaluation and required results** 

The group must evaluate the final models on the held-out test set and compare them using appropriate classification metrics. 

At minimum, the group must report the following: 

- Test accuracy. 

- Precision, recall, and F1-score. 

- Macro-averaged F1-score for multiclass or imbalanced data. 

- Per-class precision, recall, and F1-score. 

- A confusion matrix. 

- Training and validation loss curves. 

- A validation accuracy or validation F1-score curve. 

Balanced accuracy, ROC-AUC, precision-recall curves, calibration, inference time, and model size may be included when relevant, but they are not compulsory for every topic. 

The results section must go beyond listing scores. Students must identify important classification errors, show several correct and incorrect predictions, discuss the effects of class imbalance or data quality, and explain plausible reasons for performance differences among the three models. 

## **9 Project report requirements: 20 marks** 

The final written submission must be a concise project report rather than a full conference article. A length of approximately 10 to 15 pages is recommended, excluding the references and optional appendices. The report may be prepared in LaTeX or Word and must be submitted as a PDF. 

The recommended report structure is as follows: 

4 

1. The title, group members, and a short abstract. 

2. The introduction and problem motivation. 

3. The dataset source, classes, preparation, and ethical considerations. 

4. The methodology, selected models, and transfer learning approach. 

5. The experimental setup and hyperparameters. 

6. The results, figures, and model comparison. 

7. The discussion, limitations, and error analysis. 

8. The conclusion. 

9. The references. 

The writing must be clear and original. Dataset sources, pretrained models, software, diagrams, and technical claims must be acknowledged using suitable academic or official references. 

## **10 Oral presentation requirements: 10 marks** 

Each group must deliver a short oral presentation. The presentation date, duration, and venue will be communicated by the lecturer. 

All group members must participate and must be prepared to answer questions about any part of the project. 

The presentation must cover the following: 

- The selected problem and why it is important. 

- The dataset and class definitions. 

- The three selected models and the transfer learning strategy. 

- The experimental setup. 

- The main results and confusion matrix. 

- Important errors, limitations, and lessons learned. 

- The final conclusion and recommended model. 

## **11 Required submission items** 

Each group must submit the following by the announced deadline: 

- The final project report as a PDF. 

- The presentation slides. 

- One or more Jupyter Notebooks with visible outputs, figures, and metric tables. 

5 

- The Python source files or scripts used for data preparation and evaluation, where applicable. 

- A README file containing the dataset source, setup instructions, and the steps required to reproduce the work. 

- The training, validation, and test split files, or a script that recreates the split. 

- A repository link or a single organised submission archive, as instructed on eFundi. 

Large datasets and model weights do not need to be uploaded directly when this is impractical. However, the submission must provide clear access or download instructions where the licence permits this. 

## **12 Assessment rubric** 

The project is assessed out of 30 marks. The written project report contributes 20 marks, and the oral presentation contributes 10 marks. 

### **12.1 Project report rubric: 20 marks** 

|**Criterion**|**Description**|**Marks**|
|---|---|---|
|Problem defnition and<br>motivation|The classifcation problem, project aim, scope, mo-<br>tivation, and target classes are clearly defned.|2|
|Dataset<br>selection<br>and<br>preparation|The data source, class defnitions, cleaning, la-<br>belling, class distribution, data split, leakage pre-<br>vention, ethics, and licensing are explained clearly.|4|
|Model<br>selection<br>and<br>methodology|Three suitable model families are selected.<br>The<br>neural network methods,<br>transfer learning ap-<br>proach, and complete workfow are explained cor-<br>rectly.|4|
|Experimental setup and<br>reproducibility|The preprocessing, augmentation, hyperparame-<br>ters, software, hardware, random seed, and fairness<br>of the comparison are reported clearly.|3|
|Results and evaluation|The required classifcation metrics, fgures, confu-<br>sion matrix, learning curves, and comparison of all<br>three models are presented correctly.|4|
|Discussion, error analy-<br>sis, and conclusion|The results are interpreted meaningfully. Classif-<br>cation errors, limitations, and possible causes are<br>discussed, and the conclusion is justifed.|2|
|Report quality and ref-<br>erences|The report is well structured, clearly written, read-<br>able, and supported by appropriate references.|1|
|**Total project report**<br>**mark**||**20**|



**12.2 Oral presentation rubric: 10 marks** 

6 

|**Criterion**|**Description**|**Marks**|
|---|---|---|
|Problem<br>and<br>project<br>overview|The problem, aim, target classes, and practical rel-<br>evance are explained clearly.|2|
|Dataset and methodol-<br>ogy|The data preparation, three selected model fami-<br>lies, and transfer learning approach are explained<br>accurately.|2|
|Results and visual com-<br>munication|The main metrics, confusion matrix, and compar-<br>ison of the three models are presented clearly and<br>correctly.|2|
|Critical<br>understanding<br>and discussion|The group explains classifcation errors, limita-<br>tions, model behaviour, and the fnal recommen-<br>dation.|2|
|Delivery, participation,<br>and questions|The presentation is organised and professional. All<br>members contribute and answer questions compe-<br>tently.|2|
|**Total oral presenta-**<br>**tion mark**||**10**|



## **13 Final note** 

A successful project will demonstrate that the group can define a suitable classification problem, prepare and understand the data, apply neural networks and transfer learning correctly, compare three models fairly, interpret the results critically, and communicate the work clearly in a report and oral presentation. 

7 

