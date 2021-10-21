import numpy as np
import pandas as pd 
import ast
from sklearn.model_selection import train_test_split
from typing import Union, List, Dict, Tuple
from tqdm import tqdm


def convert_str_lists_to_real_lists(df: pd.DataFrame, 
                                    columns: Union[List, Tuple]) -> pd.DataFrame:
    """
    Converts string representations of lists into real lists in the certain columns.

    Args:

        df (pd.DataFrame): pandas dataframe to reformat.

        columns (tuple or list): optional, defaults to None. 
            List containing names of columns which has to be checked 
            and reformated. If not passed, all the columns will be
            checked and reformated if it's possible.

    Returns:
        df (pd.DataFrame): The return reformatted dataframe.

    Example:
        "[1, 2, 3]" (str) -> [1, 2, 3] (list)
    """

    if columns is None:
        columns = df.columns

    for column in columns:
        new_values = list()

        for value in df[column].values:

            if type(value) is not str:
                continue
            if value[0] != "[" or value[-1] != "]":
                continue

            try: 
                new_value = ast.literal_eval(new_value)
            except ValueError:
                continue

            new_values.append(new_value)

        df[column] = new_values

    return df


def get_stratified_crop(df: pd.DataFrame, size: int,
                        target_column: str, seed: int) -> pd.DataFrame:
    """
    Crops the dataframe according the selected size 
    with saving source class distibution.

    Args:

        df (pd.DataFrame): pandas dataframe to crop.

        size (int): crop size ratio.

        target_column (str): data is split in a stratified fashion,
            using this column of the dataframe as the class labels.

        seed (int): random seed.

    Returns:
        df (pd.DataFrame): The return cropped dataframe.
    """

    df, _ = train_test_split(df, train_size=size, test_size=1-size,
                             stratify=df[target_column], random_state=seed)

    return df


def get_balanced_crop(df: pd.DataFrame, target_column: str,
                      seed: int) -> pd.DataFrame:
    """
    Crops the dataframe and gets rid of class imbalance
    by taking the number of samples of each class which is
    equaled the number of samples of the least frequent class.

    Args:

        df (pd.DataFrame): pandas dataframe to crop.

        target_column (str): column with the labels
            that has to be equaled.

        seed (int): random seed.

    Returns:
        df (pd.DataFrame): The return cropped and balanced dataframe.
    """
    
    columns = df.columns
    new_df = pd.DataFrame(columns=columns)

    number = np.min(df[target_column].value_counts())
    keys = df[target_column].value_counts().keys()

    for key in keys:
        one_class_df = df[df[target_column] == key]
        samples = one_class_df.sample(n=number, random_state=seed).reset_index(drop=True)
        new_df = pd.concat([new_df, samples], ignore_index=True)

    # Shuffling
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return new_df


def oversampling(df: pd.DataFrame, target_column: str, oversampling_type: str, 
                 seed: int, downsample_higher_values: bool,
                 class_value: str, value: int) -> pd.DataFrame:
    """
    Oversamples samples for some classes in the dataframe
    up to the certain value.

    Args:

        df (pd.DataFrame): pandas dataframe to oversample.

        target_column (str): column with the labels
            which oversampling will be based on.

        oversampling_type (str): one of three types of oversampling.
        
            1) up_to_max - all the samples of each class
                will be oversampled up to the number of the
                most frequent class.

            2) by_class_value - all the samples of each class
                will be oversampled up to the number of the
                certain class. The name of this class has to be
                passed in the class_value parameter.

            3) by_value - all the samples of each class
                will be oversampled up to the custom number.
                This value has to be passed in the value parameter.

        class_value (str): used with the oversampling_type="by_class_value".
            class which frequency will be used for oversampling.

        value (int): used with the oversampling_type="by_value".
            value which all the classes will be upsampled to.

        seed (int): random seed.

        downsampling_higher_values (bool):
            If set to True, in case if oversampling_type is set to "by_class_value"
            or "by_value" and class_value or value parameter isn't equal
            to the number of samples of the most frequent class,
            the classes with the greater number of samples will be
            downsampled to this value.

    Returns:
        df (pd.DataFrame): the return oversampled dataframe.
    """

    columns = df.columns
    classes = df[target_column].unique()
    counts = df[target_column].value_counts()
    new_df = pd.DataFrame(columns=columns)

    # UP TO MAX
    if oversampling_type == "up_to_max":
        number = np.max(counts)

        for c in classes:
            one_class_df = df[df[target_column] == c]

            samples = one_class_df.sample(n=counts[c], random_state=seed).reset_index(drop=True)
            for _ in range(number // counts[c]):
                new_df = pd.concat([new_df, samples], ignore_index=True) 
            rest_samples = one_class_df.sample(n=number % counts[c],
                                               random_state=seed).reset_index(drop=True)
            new_df = pd.concat([new_df, rest_samples], ignore_index=True)

    # BY CLASS VALUE or BY CUSTOM VALUE
    else:
        number = counts[class_value] if oversampling_type == "by_class_value" else value

        for c in tqdm(classes):
            one_class_df = df[df[target_column] == c]

            if counts[c] > number:
                if downsample_higher_values == True:
                    samples = one_class_df.sample(n=counts[c], random_state=seed).reset_index(drop=True)
                    new_df = pd.concat([new_df, samples], ignore_index=True)
                else:
                    samples = df[df[target_column] == c]
                    new_df = pd.concat([new_df, samples], ignore_index=True)

            else:      
                samples = one_class_df.sample(n=counts[c], random_state=seed).reset_index(drop=True)
                for _ in range(number // counts[c]):
                    new_df = pd.concat([new_df, samples], ignore_index=True) 
                rest_samples = one_class_df.sample(n=number % counts[c],
                                                random_state=seed).reset_index(drop=True)
                new_df = pd.concat([new_df, rest_samples], ignore_index=True)

    # Shuffling
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return new_df 


df = pd.read_csv("/home/toefl/K/GLR/landmark-recognition-2021/train.csv")
df = oversampling(df, "landmark_id", "by_value", 0xFACED, False, None, 5)
df.to_csv("oversampled.csv")