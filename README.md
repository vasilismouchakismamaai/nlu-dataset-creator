# nlu-dataset-creator

## create intents dataset based on custom format
```
python -m create-datasets intents -p data.csv -s ',' -e csv -o data -d yes
```
## create entities dataset based on custom format
```
python -m create-datasets entities -p data.csv -s ',' -e csv -o data -t manual
```
## create entities dataset based on doccano format
```
python -m create-datasets entities -p admin.json -s ',' -e csv -o data -t doccano
```