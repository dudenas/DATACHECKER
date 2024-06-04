#!/usr/bin/env python
# coding: utf-8

# In[19]:


import shutil
from IPython.display import FileLink
import pandas as pd
import re
from IPython.display import display
from ipyfilechooser import FileChooser
from tqdm import tqdm
import os
import numpy as np
import sys
from tabulate import tabulate

from datetime import date

from prettytable import PrettyTable
from pathlib import Path

import json

# ignore group warnings
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression, and has match groups')

# Add complete filepath here.
filepath = sys.argv[1] if len(sys.argv) > 1 else ''
# filepath = "datasets/AusLCIConstruction_MVP02_EF_20240514.csv"
zip = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False

filename = Path(filepath).stem
folder_to_download = filename  # Use the same name as the CSV file you are feeding in. At the very bottom of the page, once the report is generated, a Zip file becomes available to download.


# In[20]:


# Open the JSON file and load columns into variables
with open("columns.json", 'r') as json_file:
    # Load the JSON data into a Python object
    data = json.load(json_file)
    universal_tabular_columns = data["universal_tabular_columns"]
    non_discerning_columns = data["non_discerning_columns"]


# In[21]:


# change if there is an error reading csv or it is strangely formatted
# df = pd.read_csv(filepath, on_bad_lines='skip')
df = pd.read_csv(filepath)
# df = pd.read_csv(filepath, sep='delimiter', header=None)

# add index to match the csv
df.index += 2

# print(df.columns)

# set current date
today = date.today()

# YYmmdd
CURRENT_DATE = today.strftime("%Y%m%d")

# CREATE FOLDERS
# create a folder for each document
FOLDER_NAME = re.sub("[^A-Z0-9]", " ", filename,0,re.IGNORECASE)
FOLDER_NAME = re.sub(' +', '_', FOLDER_NAME)

FOLDER_PATH = os.path.join(os.getcwd(), FOLDER_NAME)
print(f'Folder path : {FOLDER_PATH}')
if not os.path.exists(FOLDER_PATH):
    os.mkdir(FOLDER_PATH)

# formatting
if not os.path.exists(f"{FOLDER_PATH}/FORMATTING"):
    os.mkdir(f"{FOLDER_PATH}/FORMATTING")

# repetition
if not os.path.exists(f"{FOLDER_PATH}/REPETITION"):
    os.mkdir(f"{FOLDER_PATH}/REPETITION")

# other
if not os.path.exists(f"{FOLDER_PATH}/OTHER"):
    os.mkdir(f"{FOLDER_PATH}/OTHER")

# PRINT TABLE
MASTER_TABLE = ""
CMD_TABLE= []
def add_to_cmd_table(name, value):
    global CMD_TABLE
    CMD_TABLE.append([name, value])

def print_table(table, horizontal, column_headers = []):
    global MASTER_TABLE
    tab = None
    if horizontal:
        tab = PrettyTable(table[0])
        tab.add_rows(table[1:])
    else:
        tab = PrettyTable()
        for index, item in enumerate(column_headers):
            tab.add_column(item, table[index])
    tab.padding_width = 2
    tab.junction_char = '.'
    # print(tab)
    MASTER_TABLE += tab.get_csv_string()
    MASTER_TABLE += "-\n"
    MASTER_TABLE += "-\n"
    MASTER_TABLE += "-\n"

# print(df.iloc[df.shape[0]-1])
# print(df.shape[0])
# # Drop rows with any empty cells // put more threshold to kick the strange ones
# df.dropna(how='all',thresh=1, inplace=True)
# print(df.shape[0])

MASTER_TABLE += "Total data points\n"
MASTER_TABLE += f"{df.shape[0]}\n"
MASTER_TABLE += "-\n"
MASTER_TABLE += "-\n"
MASTER_TABLE += "-\n"

# print(len(df.columns))


# In[22]:


# check whether existing columns are in the same names as universal tabular columns
for index, item in enumerate(df.columns):
    if item not in universal_tabular_columns:
        print(f'this is not the right name : {item}')

# add missing columns 
for index, item in enumerate(universal_tabular_columns):
    if item not in df.columns:
        print(f'the column was added : {item}')
        # df[item] = pd.Series()
        df[item] = pd.Series(dtype='object')


# In[23]:


def duplicates_non_discerning(dataFrame, df_columns, table_text, analyse_name):
    global MASTER_TABLE
    # finds all duplicate rows and keeps all of them
    df_local = dataFrame[df_columns]
    df_range = df_local[df_local.duplicated(keep=False)]

    # sort based on the first column
    df_range = df_range.sort_values(by=[df_columns[0]])

    # total dataframe count
    df_count = dataFrame.shape[0]
    
    # Number of instances
    df_instance_count = df_range.shape[0]

    # Number of records that would be removed
    df_without_range = df_local.drop_duplicates()
    df_range_count = df_count - df_without_range.shape[0]

    # - % of total data affected (copies/total)
    df_range_percent = df_range_count / df_count

    # print(f"{analyse_name}")
    MASTER_TABLE += f"{analyse_name}\n"

    # if there are no values exit
    if df_range_percent == 0:
        MASTER_TABLE += "PASS\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        add_to_cmd_table(analyse_name, "PASS")
        return
    add_to_cmd_table(analyse_name, "FIX")
    
    table = [table_text, [df_instance_count, df_range_count, df_range_percent]]
    print_table(table, True)

    # SAVE FILES
    # save - Rows that are duplicated - OUTPUT 1
    df_range_full_table = dataFrame.loc[df_range.index]
    df_range_full_table.to_csv(f'{FOLDER_PATH}/OTHER/Report-{analyse_name}-Output_1-{filename}-{CURRENT_DATE}.csv',encoding='utf-8')
    
    # print(f"Saved — {analyse_name}-Report-Output_1")

    # save - Duplicate instances & number of each - OUTPUT 2
    # count the duplicates
    df_range_group = df_range.groupby(df_range.columns.tolist(),as_index=False, dropna=False, sort=True).size()
    # add count to begining
    df_range_group.insert(0, "count", df_range_group["size"])
    # remove size at the end
    df_range_group.pop('size')
    # sort values by count
    df_range_group = df_range_group.sort_values(by="count", ascending=False)
    # reset index
    df_range_group = df_range_group.reset_index(drop=True)
    # save
    df_range_group.to_csv(f'{FOLDER_PATH}/OTHER/Report-{analyse_name}-Output_2-{filename}-{CURRENT_DATE}.csv',encoding='utf-8')

# DUPLICATES
table_text = ['Number of instances', 'Number of records that would be removed', '% of total data affected (copies/total)']
duplicates_non_discerning(df, df.columns, table_text, "Duplication")

# drop duplicates
df = df.drop_duplicates()

# NON DISCERNING-INFO
table_text = ['Number of instances', 'Number of records that indiscernable', '% of rows of the total data affected (copies/total)']
duplicates_non_discerning(df, non_discerning_columns, table_text, "Non_discerning_info")


# In[25]:


def missing_value(dataFrame):
    global MASTER_TABLE

    # total dataframe count
    df_count = dataFrame.shape[0]

    def change_missing_values(mvv):
        mvv_changed = []
        for i in mvv:
            val = i
            if i == 0:
                val = "full"
            elif i == df_count:
                val = "---"
            mvv_changed.append(val)
                
        return mvv_changed

    df_mandatory_ef_columns = ["Name", "Value", "Unit"] # Emission Factors
    df_mandatory_c_columns = ["Name", "Components"] # Compositions

    # check whether it is emmission factor or composition
    if dataFrame["Value"].isna().sum() == df_count:
        df_mandatory_columns= df_mandatory_c_columns
    else:
        df_mandatory_columns= df_mandatory_ef_columns
    df_optional = dataFrame.drop(columns = df_mandatory_columns) # mandatory dataframe
    df_mandatory = dataFrame[df_mandatory_columns] # mandatory dataframe
    
    # MANDATORY
    missing_value_table = df_mandatory.isna().sum()
    missing_value_values = change_missing_values(missing_value_table.tolist())
    
    # save misisng value table
    MASTER_TABLE += f"Missing_value_mandatory\n"

    # if it does not miss any values
    if len([x for x in missing_value_values if x != "full"]) == 0:
        MASTER_TABLE += "PASS\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        add_to_cmd_table("Missing_value_mandatory", "PASS")
    else:
        table = [missing_value_table.index.tolist(), missing_value_values]
        print_table(table, False, ["Column", "Count"])
        add_to_cmd_table("Missing_value_mandatory", "FIX")

        # go trough each column in the dataFrame and check which columns misses somve values. If it does, save the document
        for col_name, val in missing_value_table.items():
            if val != dataFrame.shape[0] and val != 0:
                cn = re.sub("[^A-Z]", " ", col_name,0,re.IGNORECASE)
                cn = re.sub(' +', '_', cn)
                fname = f'{FOLDER_PATH}/OTHER/Report-Missing_value_mandatory-{cn}-Output_1-{filename}-{CURRENT_DATE}.csv'
                df_isna = dataFrame[dataFrame[col_name].isna()]
                df_isna.to_csv(fname,encoding='utf-8')
    
    # OPTIONAL
    missing_value_table = df_optional.isna().sum()
    missing_value_values = change_missing_values(missing_value_table.tolist())
    
    # save misisng value table
    MASTER_TABLE += f"Missing_value_optional\n"

    # if it does not miss any values
    table = [missing_value_table.index.tolist(), missing_value_values]
    print_table(table, False, ["Column", "Count"])

    # go trough each column in the dataFrame and check which columns misses somve values. If it does, save the document
    for col_name, val in missing_value_table.items():
        if val != dataFrame.shape[0] and val != 0:
            cn = re.sub("[^A-Z]", " ", col_name,0,re.IGNORECASE)
            cn = re.sub(' +', '_', cn)
            fname = f'{FOLDER_PATH}/OTHER/Report-Missing_value_optional-{cn}-Output_1-{filename}-{CURRENT_DATE}.csv'
            df_isna = dataFrame[dataFrame[col_name].isna()]
            df_isna.to_csv(fname,encoding='utf-8')


missing_value(df)


# In[26]:


def repetition(dataFrame, df_columns, table_text, analyse_name):
    global MASTER_TABLE
    # finds all duplicate rows and keeps all of them
    df_local = dataFrame[df_columns]
    df_range = df_local[df_local.duplicated(keep=False)]

    # sort based on the first column
    df_range = df_range.sort_values(by=[df_columns[0]])

    # total dataframe count
    df_count = dataFrame.shape[0]

    # count the duplicates
    df_range_group = df_range.groupby(df_range.columns.tolist(),as_index=False, dropna=False, sort=True).size()
    # sort values by count
    df_range_group = df_range_group.sort_values(by="size", ascending=False)
    # reset index
    df_range_group = df_range_group.reset_index(drop=True)

    # print 
    MASTER_TABLE += f"{analyse_name}\n"
    MASTER_TABLE += f"{table_text}\n"

    # if there are no values exit
    if df_range_group.shape[0] == 0:
        MASTER_TABLE += "PASS\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        add_to_cmd_table(analyse_name, "PASS")
        return
    
    add_to_cmd_table(analyse_name, "FIX")

    # go trough the range and add it to the list
    df_range_vals = [2,5,10,25,50,100,1000,10000]
    df_total = []
    df_instances = []
    df_percent = []
    for index, value in enumerate(df_range_vals):
        partition_of_data = df_range_group[df_range_group["size"] >= value]
        partition_of_data_total = partition_of_data.sum()["size"]
        partition_of_data_instances = len(partition_of_data["size"])
        partition_of_data_percent = partition_of_data_total/df_count

        df_total.append(partition_of_data_total)
        df_instances.append(partition_of_data_instances)
        df_percent.append(partition_of_data_percent)

    table = [df_range_vals,df_total,df_instances,df_percent]
    print_table(table, False, ["more or equal to copies ", "data affected", "instance count", "percent"])


    # SAVE FILES
    # save - Rows that are duplicated - OUTPUT 1
    df_range_full_table = dataFrame.loc[df_range.index]
    df_range_full_table.to_csv(f'{FOLDER_PATH}/REPETITION/Report-{analyse_name}-Output_1-{filename}-{CURRENT_DATE}.csv',encoding='utf-8')

    # save - Duplicate instances & number of each - OUTPUT 2
    # count the duplicates
    df_range_group = df_range.groupby(df_range.columns.tolist(),as_index=False, dropna=False, sort=True).size()
    # add count to begining
    df_range_group.insert(0, "count", df_range_group["size"])
    # remove size at the end
    df_range_group.pop('size')
    # sort values by count
    df_range_group = df_range_group.sort_values(by="count", ascending=False)
    # reset index
    df_range_group = df_range_group.reset_index(drop=True)
    # save
    df_range_group.to_csv(f'{FOLDER_PATH}/REPETITION/Report-{analyse_name}-Output_2-{filename}-{CURRENT_DATE}.csv',encoding='utf-8')

# REPETITION 1
table_text = 'The Name instance'
repetition(df, ["Name"], table_text, "Repetition_1")

# REPETITION 2
table_text = 'The Description instance'
repetition(df, ["Description"], table_text, "Repetition_2")

try:
    # if (df[["Components"]].isna().sum() == df.shape[0]).bool() == False:
    if (df[["Components"]].isnull().sum() == df.shape[0]).all() == False:
        # REPETITION 3
        table_text = 'The Comp. Items List instance'
        repetition(df, ["Components"], table_text, "Repetition_3")
    else:
        # REPETITION 4
        table_text = 'The Value instance'
        repetition(df, ["Value"], table_text, "Repetition_4")
except:
    print("The dataset does not include all columns")

    # REPETITION 4
    table_text = 'The Value instance'
    repetition(df, ["Value"], table_text, "Repetition_4")
    pass


# In[27]:


def format_1(dataFrame, df_columns, analyse_name):
    global MASTER_TABLE
    
    def check_spelling(item):
        try:
            row = item[df_columns[0]]
            row_split = row.split()
            for i in row_split:
                if not i[0].isupper():
                    if not i == "of" and not i == "and" and not i =="the":
                        return True
            return False
        except:
            print(f"Exception, inspect it : {item}")
            return True

    df_local = dataFrame[df_columns]

    df_results = df_local[df_local.apply(check_spelling, axis=1)]

    # check upper case
    count = df_results.shape[0]
    indexes = df_results.index

    # count percent
    df_percent = count / df_local.shape[0]

    # print table
    MASTER_TABLE += f"{analyse_name}\n"
    
    # if there are no values exit
    if df_percent == 0:
        MASTER_TABLE += "% of rows where the Region/Regional Scope column items is without capitalization\n"
        MASTER_TABLE += "PASS\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"

        add_to_cmd_table(analyse_name, "PASS")
        return
    
    add_to_cmd_table(analyse_name, "FIX")
    
    table = [["% of rows where the Region/Regional Scope column items is without capitalization"],[df_percent]]
    print_table(table, True)

    # save document
    df_full_table = dataFrame.loc[indexes]
    df_full_table.to_csv(f'{FOLDER_PATH}/FORMATTING/Report-{analyse_name}-Output_1-{filename}-{CURRENT_DATE}.csv',encoding='utf-8')


if (df[["Region/Regional Scope"]].isnull().sum() == df.shape[0]).all() == False:            
    format_1(df, ["Region/Regional Scope"], "Formatting-1")
else:
    print("No Region/Regional Scope")
    # print table
    MASTER_TABLE += "Formatting_1\n"
    MASTER_TABLE += "% of rows where the Region/Regional Scope column items is without capitalization\n"
    MASTER_TABLE += "No Region/Regional Scope \n"
    MASTER_TABLE += "-\n"
    MASTER_TABLE += "-\n"
    MASTER_TABLE += "-\n"
    


# In[28]:


def format_2(dataFrame, df_columns, analyse_name):
    global MASTER_TABLE
    
    def check_numbers(item):
        row = item[df_columns[0]]
        if not isinstance(row,float) and not isinstance(row,int) and not isinstance(row, np.int64):
            return True
        else:
            return False
    
    df_local = dataFrame[df_columns]

    df_results = df_local[df_local.apply(check_numbers, axis=1)]

    # check upper case
    count = df_results.shape[0]
    indexes = df_results.index

    # count percent
    df_percent = count / df_local.shape[0]

    # print table
    MASTER_TABLE += f"{analyse_name}\n"
    
    # if there are no values exit
    if df_percent == 0:
        MASTER_TABLE += "% of rows where the Value column does items not contain a number\n"
        MASTER_TABLE += "PASS\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"

        add_to_cmd_table(analyse_name, "PASS")
        return
    
    add_to_cmd_table(analyse_name, "FIX")

    table = [["% of rows where the Value column does items not contain a number"],[df_percent]]
    print_table(table, True)


    # save document
    df_full_table = dataFrame.loc[indexes]
    df_full_table.to_csv(f'{FOLDER_PATH}/FORMATTING/Report-{analyse_name}-Output_1-{filename}-{CURRENT_DATE}.csv',encoding='utf-8')
            
format_2(df, ["Value"], "Formatting_2")


# In[29]:


def format_all_columns(dataFrame, df_columns, regex, analyse_name, description_text):
    global MASTER_TABLE

    # drop source as it always has double characters
    df_local = dataFrame[df_columns].drop("Source", axis=1)

    indexes = []
    output_count = []

    # go trough columns and check whether in any of those there is a questionmark
    for index, item in enumerate(df_local.columns):
        # stringify the column
        # temp = dataFrame[item].astype(str).str.replace('.', '', regex=True)
        temp = dataFrame[item].astype(str)
        search_range = temp[temp.str.contains(regex)]
        
        # set indexes to search range ndexes
        indexes = search_range.index.tolist()

        # append to output count
        output_count.append(str(len(indexes)))

        # save file if found
        if len(indexes) > 0:
            # create a file name
            cn = re.sub("[^A-Z]", " ", item,0,re.IGNORECASE)
            cn = re.sub(' +', '_', cn)

            # save document
            df_full_table = dataFrame.loc[indexes]
            df_full_table.to_csv(f'{FOLDER_PATH}/FORMATTING/Report-{analyse_name}-{cn}-Output_1-{filename}-{CURRENT_DATE}.csv',encoding='utf-8')

    # calculate percentages
    output_percentages = [int(i) / df_local.shape[0] for i in output_count]

    # print table
    MASTER_TABLE += f"{analyse_name}\n"
    MASTER_TABLE += f"{description_text}\n"

    # if there are no values exit
    if  len([x for x in output_count if int(x) != 0]) == 0:
        MASTER_TABLE += "PASS\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        MASTER_TABLE += "-\n"
        add_to_cmd_table(analyse_name, "PASS")
        return
    
    add_to_cmd_table(analyse_name, "FIX")
    
    table = [df_local.columns.tolist(), output_percentages, output_count]
    print_table(table, False, ["Column", "Percentage", "Count"])

regex = r"([^A-Za-z0-9 ])\1"
description_text = "% of rows with a doubled character that is not a letter or number in any column & count"
format_all_columns(df, df.columns, regex, "Formatting_3", description_text)

regex = r"([ ])\1"
description_text = "% of rows with a doubled space in any column & count"
format_all_columns(df, df.columns, regex, "Formatting_4",description_text)

regex = r"([?])"
description_text = "% of rows with a question mark in any column & count"
format_all_columns(df, df.columns, regex, "Formatting_5",description_text)


# In[30]:


def save_table ():
    global MASTER_TABLE
    fname = f'{FOLDER_PATH}/Report-{filename}-{CURRENT_DATE}.csv'
    with open(f'{fname}', 'w', newline='') as f_output:
        f_output.write(MASTER_TABLE)
save_table()


# In[31]:


def split_files(directory):
    # Go through directory and split files
    # fname = f'{FOLDER_PATH}/Report-{filename}-{CURRENT_DATE}.csv'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # check whether the file ending is csv
            if f[-3:] == "csv":
                # print(f)
                # check size of the file
                if os.path.getsize(f) > 50000000:
                    print(f'splitting : {filename}')
                    # make chunk out of big page
                    chunkFolder = f"{directory}/{filename[:-4]}/"
                    if not os.path.exists(chunkFolder):
                        os.mkdir(chunkFolder)
                    for i,chunk in enumerate(pd.read_csv(f, chunksize=250000)):
                        chunk.to_csv(chunkFolder+f'{filename[:-4]}'+'-{}.csv'.format(i), index=False)
                        print(i)
                    # remove file from the system
                    os.remove(f)
        if os.path.isdir(f):
            split_files(f)
split_files(FOLDER_PATH)


# In[32]:


# Only make an archive if zip is True
if zip:
    shutil.make_archive(folder_to_download, 'zip', folder_to_download)
    filelink = folder_to_download+'.zip'
    FileLink(filelink)


# In[ ]:


# Column headers
headers = ["Name", "Value"]
colalign = ("left", "left")
print(tabulate(CMD_TABLE, headers, tablefmt="pretty", colalign=colalign))


# In[ ]:




