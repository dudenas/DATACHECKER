#!/bin/sh

# Initialize variables for the options
filepath=""
zip=""

# Parse the options
while getopts ":f:z:h" option
do
case "${option}"
in
f) filepath=${OPTARG};;
z) zip="true";;
h) echo "Usage: $0 -f filepath [-z]"
   echo "-f filepath: The path to the file to be checked."
   echo "-z: Optional argument. Pass to enable zip."
   exit 0;;
\?) echo "Invalid option: -$OPTARG" >&2
   exit 1;;
esac
done

# Check if a file path is provided
if [ -z "$filepath" ]; then
    echo "File path not provided. Usage: $0 -f filepath [-z]"
    exit 1
fi

# Run the Python script with the file path as an argument, and the boolean as an optional argument
if [ -n "$zip" ]; then
    python3 datachecker.py "$filepath" "$zip"
else
    python3 datachecker.py "$filepath"
fi