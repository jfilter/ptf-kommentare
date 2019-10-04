## Instructions for data preprocessing for (dynamic) Bernoulli embeddings.

Preprocess the data with 3 simple steps described below.

Depending on how large your data is, you should set aside ca. 20 minutes for these preprocessing steps. But they are worth it. They make running the algorithms significantly faster. The other good news is that code is included for all the preprocessing steps.

The goal of preprocessing is to truncate the vocabulary, remove all the words that are not in the vocabulary and then put the text into numpy arrays. Then we split the data into training validation and test set and compute statistics of the data that we need to use in the algorithm (e.g the number of words in each time bin, and the prefixes of file names that hold the data for each time bin).

These are the steps. They are described in detail below:

  0. Decide on filenames for time slices
  1. Create vocabulary file and save text data in numpy arrays
  2. Subsample the data and split into training and testing
  3. Create `dat_stats.pkl`


### Reqired Input

The following folder structure is required to run dynamic Bernoulli embeddings.

#### Before Preprocessing:
Assumption: Your text is in text files in the `dat/[dataset_name]/raw/` subfolder.

```
dat/
    [dataset_name]/
        raw/
            *.txt
```

#### After Preprocessing:

The preprocessing scripts will add the following folders and files 
```
dat/
    [dataset_name]/
        unigram.txt
        dat_stats.pkl
        raw/
            *.txt
            *.npy
        train/
            *.npy
        test/
            *.npy
        valid/
            *.npy
```            

The `train/`, `test/` and `valid/` folders will contain the `.npy` files with the data.
The file `unigram.txt` contains the vocabulary and the vocabulary counts and the file `dat_stats.pkl` contains a pickle object that hold a python dictionary with information about the data required to run the algorithm.


  
 ### 0. Decide on filenames for the time slices
 
Your text files are in the `txt/` subfolder. Decide now how many time slices you want and give each time slice its own prefix.
I like using the dates as prefix. So say you have data for the years 1990 - 2000 and want each year to be a time slice. You can make the prefixes `[90, 91, 92, ..., 00]`.
Then make sure that each filename starts with one of the prefixes depending on which time slice it belongs to.

i.e. all files with text data from 1992 start with `92` and have a name like `92_rest_of_file_name_does_not_matter.txt` or `92_record_1.txt`.
Save the list of prefixes `[90, 91, 92, ..., 00]`. You will need it in step 4.

Tip: Instead of having many, many short files. You might want to have few longer files in each time slice. Simply concatenate multiple short files into longer text files.

### 1. Create vocabulary file and save text data in numpy arrays

In this step you will run the script `step_1_count_words.py`.
The script counts distinct words and truncates the resulting vocabulary.
Then each word is replaced whith its index in the vocabulary and the resulting numpy arrays are saved.

#### 1.1 Modify step_1_count_words.py

Go into this file and change `dataset_name` to the name of the folder in which the data is.
It should be the subfolder unted `dat/` (as specified above your data is in `dat/[dataset_name]/raw/`)

#### 1.2 Create directories for training testing and valid

            ```
              mkdir [dataset_name]/train
              mkdir [dataset_name]/test
              mkdir [dataset_name]/valid
            ```
            
Make sure to replace `[dataset_name]` with the dataset name you used in step 1.1.

#### 1.3 Run step_1_count_words.py
Assuming you are in `dat/` simply run 

```
   python step_1_count_words.py
```

Tip: In this script data preprocesing is handled. Punctuation is removed, line-breaks are handled and everything is lower cased. Depending on your dataset, additional or differnt preprocessing steps might be required. We also recommend extracting bigrams and addind them to the vocabulary.

### 2. Sumsample the data and split into train test and validation split

As in step 1.1 open the file `step_2_split_data.py` and change the data_set name.
Then form `dat/` run
```
    python step_2_split_data.py
```

### 3. Create data statistics.

In this step, you simply have to run `step_3_create_data_stats.py`.
Again, go into the file and change the `dataset_name`. 
Then add the list of time slices from step 0 under `time_slices`
(e.g. `time_slices = [90, 91, 92, ..., 00]`) and add the prefix length (in this case `prefix_length=2`).
You also have the option to add query words. For these words the algorithm will print out the dynamic embeddings.
More information is in the comments in `step_3_create_data_stats.py`.

Finally, run
```
    python step_3_create_data_stats.py
```
