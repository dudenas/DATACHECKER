#!/bin/sh

# Initialize variables for the options
filepath=""
zip=""
remove="true"

# Parse the options
while getopts ":f:z:rh" option
do
case "${option}"
in
f) filepath=${OPTARG};;
z) zip="true";;
r) remove="";;
h) echo "Usage: $0 -f filepath [-z] [-r]"
   echo "-f filepath: The path to the file to be checked."
   echo "-z: Optional argument. Pass to enable zip."
   echo "-r: Optional argument. Pass to prevent removal of the directory."
   exit 0;;
\?) echo "Invalid option: -$OPTARG" >&2
   exit 1;;
esac
done

# Check if a file path is provided
if [ -z "$filepath" ]; then
    echo "File path not provided. Usage: $0 -f filepath [-z] [-r]"
    exit 1
fi

# Run the Python script with the file path as an argument, and the boolean as an optional argument
if [ -n "$zip" ]; then
    python3 datachecker.py "$filepath" "$zip"
else
    python3 datachecker.py "$filepath"
fi

# Remove the directory if -r is not passed
if [ -n "$remove" ]; then
    filename=$(basename "$filepath")
    filename_without_ext="${filename%.*}"
    dirpath=$filename_without_ext
    rm -rf "$dirpath"
fi