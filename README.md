To convert a jupyter notebook to running python code, run a command

```shell
jupyter nbconvert --to python dataset_analysis_concat-240604.ipynb --output datachecker.py
```

To run a comman script pass a command

Command | Comment
--- | ---
**-f** | filepath should be changed to the filepath of the file
**-zip** | is optional, if passed will zip the report


```shell
sh datachecker_run.sh -f filepath -zip
```