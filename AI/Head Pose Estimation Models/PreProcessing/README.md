
# Datapreprocessing

This repository contains scripts for preprocessing two different datasets: the BIWI dataset and the 300W-LP dataset. The processed datasets are saved on the cloud and used in the `datasets` folder during the test bench process.

## Files

1. `process_biwi.py`
2. `process_300W-lp.py`

### process_biwi.py

This script is used to preprocess the BIWI dataset. It performs various data cleaning and transformation steps to prepare the dataset for further analysis and modeling.

#### InDepth explanation
This script is designed to preprocess the BIWI dataset by performing the following steps:

1. **Setup and Initialization**: 
   - Import necessary libraries, including MTCNN for face detection and OpenCV for image processing.
   - Define file paths, image size, adjustment parameter for bounding boxes, and initialize the face detector.

2. **Collect File Names**:
   - Loop through the specified directories containing the dataset.
   - Collect and sort `.png` image files and `.txt` pose annotation files.

3. **Process Images and Annotations**:
   - Loop through each directory and each image within the directory.
   - For each image, verify matching annotation files.
   - Load and parse pose annotations from the `.txt` files, converting rotation matrices to yaw, pitch, and roll angles.
   - Read the image using OpenCV and detect faces using MTCNN.
   - Adjust the bounding boxes around detected faces and resize the face regions to the specified image size.
   - Store the processed images and corresponding pose annotations in lists.

4. **Save Processed Data**:
   - Save the processed images and pose annotations into a `.npz` file for later use.

This script ensures that the BIWI dataset is properly preprocessed and ready for use in machine learning models or other analyses.


**Usage:**

```python
python process_biwi.py
```

### process_300W-lp.py

This script is used to preprocess the 300W-LP dataset. Similar to the BIWI preprocessing script, it includes steps for data cleaning and transformation specific to the 300W-LP dataset.
#### preliminary stage
the 300W-lp dataset has to be preprocessed by the create_filename_list.py to produce the file.txt that this script uses

#### InDepth explanation
This script processes a dataset comprising image and MATLAB data files:

1. **Data Reading and Initialization:**
   - Reads a list of file names from "files.txt" into a Pandas DataFrame (`inputd`).
   - Initializes an empty Pandas DataFrame (`outputd`) to store processed data.

2. **Feature Extraction Functions:**
   - **`image_to_array(path)`**: Converts an image file (`path`) to a NumPy array after converting it to RGB format.
   - **`extract_features(path)`**: Extracts pose parameters (`yaw`, `pitch`, `roll`) from MATLAB data (`path`).
   - **`extractmin_max(path)`**: Determines minimum and maximum coordinates (`x_min`, `y_min`, `x_max`, `y_max`) from MATLAB data (`path`), incorporating a random scaling adjustment.

3. **Data Processing:**
   - Creates columns in `outputd` DataFrame:
     - "image": Stores file names from `inputd`.
     - "yaw", "pitch", "roll": Applies `extract_features` to each file in `inputd`, storing results in separate columns.
     - "x_min", "y_min", "x_max", "y_max": Applies `extractmin_max` to each file in `inputd`, storing results in separate columns.

4. **Output and Storage:**
   - Saves the processed DataFrame (`outputd`) to a pickle file named "file.pkl".
   - Prints the initial rows of `outputd` to verify processing.

In essence, the script systematically processes image and MATLAB data files, extracting specified features and coordinates, organizing them into a structured DataFrame, and persisting the processed data for further analysis or applications.

**Usage:**

```python
python process_300W-lp.py
```

## Cloud Storage

Both datasets produced from these scripts are saved on the cloud. Ensure you have the necessary credentials and access rights to store and retrieve data from your cloud storage solution.
- <a href="https://drive.google.com/file/d/1_DwVRbrPiK1y9kbBtpIJOtcD8aSu8WAX/view?usp=drive_link">file.pkl</a>
- <a href="https://drive.google.com/file/d/1vhXP7-BogBMrXjug2a4snju1_37A5D5c/view?usp=sharing">BIWI_done.npz</a>

## Test Bench Process

The processed datasets are used in the `datasets` folder on the test bench. Make sure to update the paths in your test bench scripts to point to the correct locations of the processed datasets.

## Important Remark 
process_BIWI output will produce an npz file that could be used standalone has the images with the labels
process_300W-lp output will produce an pkl file that has to be used in the root of 300W-lp dataset
