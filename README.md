To convert a jupyter notebook to running python code, run a command

```shell
jupyter nbconvert --to python dataset_analysis_concat-240604.ipynb --output datachecker.py
```

To run a comman script pass a command

Command | Comment
--- | ---
**-f** | filepath: The path to the file to be checked.
**-zip** | Optional argument. Pass to enable zip.
**-r** | Optional argument. Pass to prevent removal of the directory.


```shell
sh datachecker_run.sh -f filepath -zip -r
```