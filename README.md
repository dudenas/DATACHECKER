V2 test

### requirements
```shell
pip install -r requirements.txt
```

### jupyter notebook -> python
To convert a jupyter notebook to running python code, run a command

```shell
jupyter nbconvert --to python datachecker.ipynb --output datachecker.py
```

### run datachecker in teminal
To run a comman script pass a command

Command | Comment
--- | ---
**-f** | filepath: The path to the file to be checked.
**-zip** | Optional argument. Pass to enable zip.
**-r** | Optional argument. Pass to have a report folder.


```shell
sh datachecker_run.sh -f filepath -zip -r
```

### Example output

| Name                    | Value |
|-------------------------|-------|
| Duplication             | PASS  |
| Non_discerning_info     | PASS  |
| Missing_value_mandatory | PASS  |
| Repetition_1            | FIX   |
| Repetition_2            | FIX   |
| Repetition_4            | FIX   |
| Formatting_2            | PASS  |
| Formatting_3            | PASS  |
| Formatting_4            | PASS  |
| Formatting_5            | PASS  |
| Formatting_6            | PASS  |
| Formatting_7            | PASS  |
| Formatting_8            | PASS  |