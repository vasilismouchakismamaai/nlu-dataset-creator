import sys
import argparse
import pandas as pd
import numpy as np
import json
from pandas import json_normalize
import string
from unidecode import unidecode


def load_data(file_path: str, separator: str, extention: str) -> pd.DataFrame:
    """
    :param file_path: path to a file
    :param delimiter: delimiter for csv files
    :param extention: the extention of the file (csv/json)
    """
    if extention == "csv":
        column_names = ("text", "label", "entity-label")
        separator = separator
        data = pd.read_csv(file_path, names=column_names, delimiter=separator, keep_default_na=False)
    elif extention == "json":
        data = pd.read_json(path_or_buf=file_path, lines=True)
    else:
        sys.exit(F"The {extention} file extention is not supported")

    return data


def bio_tag(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    :param dataset: dataset to be bio-tagged
    :return: a bio-tagged dataframe
    """
    # actual bio-tagging
    w = dataset["text"].str.replace(r"[^\w\s]+", "", regex=True)
    splitted = dataset.pop("entity").str.split()
    e = dataset.pop("tag")

    m1 = splitted.str[0].eq(w)
    m2 = [b in a for a, b in zip(splitted, w)]

    dataset["tag"] = np.select([m1, m2 & ~m1], ["B-" + e, "I-" + e],  default="O")

    dataset.loc[dataset["tag"] == "B-O", "tag"] = "O"
    for index, row in dataset.iterrows():
        if row.tag == "B-O":
            row["tag"] = "O"

    return dataset


def handle_diacritics(data: pd.DataFrame, task: str):
    print(task)
    if task == "intents":
        d_text = data["text"].apply(unidecode)
        d_label = data["label"].apply(unidecode)
        d_data = pd.DataFrame({"text": d_text, "label": d_label})
        result = pd.concat([data, d_data])
        return result
    elif task == "entities":
        print(data)
        return data
    else:
        exit()


def prepare_intent_dataset(file_path: str,
                           separator: str,
                           extention: str,
                           file_name: str,
                           task: str = "intents",
                           diacritics: bool = True,
                           save: bool = True) -> pd.DataFrame:
    """
    :param file_path: path to a file
    :param delimiter: delimiter for csv files
    :param extention: the extention of the file (csv/json)
    :param file_name: name for saving the final dataset
    :param task: task, in this case 'intents
    :param diaciritcs: if yes, duplicate data without diacritics
    :param save: flag for saving the dataset if true
    :return: a dataset ready for model training
    """
    data = load_data(file_path=file_path, separator=separator, extention=extention)
    data = data[["text", "label"]]
    if diacritics:
        data = handle_diacritics(data, task)
    if save:
        outpath = f"{task}-{file_name}.csv"
        data.to_csv(outpath, header=["text", "label"], index=False)
    return data


def prepare_entities_dataset(file_path: str,
                             separator: str,
                             extention: str,
                             source_type: str,
                             file_name: str,
                             task: str = "entities",
                             diacritics: bool = True,
                             save: bool = True) -> pd.DataFrame:
    """
    :param file_path: path to a file
    :param delimiter: delimiter for csv files
    :param extention: the extention of the file (csv/json)
    :param source_type: the source of the initial dataset (manual/doccano)
    :param file_name: name for saving the final dataset
    :param task: task, in this case 'entities'
    :param diaciritcs: if yes, duplicate data without diacritics
    :param save: flag for saving the dataset if true
    :return: a dataset ready for model training
    """
    data = load_data(file_path=file_path, separator=separator, extention=extention)
    if source_type == "manual":
        data = data[["text", "entity-label"]]
        data = data.rename(columns={"index": "Sentence #"})

        data = data.reset_index()
        data = data.rename(columns={"index": "Sentence #"})
        data["Sentence #"] = "Sentence: " + data["Sentence #"].astype(str)
        data["text"] = data["text"].str.replace(r"[^\w\s]+", "", regex=True)
        data["text"] = data["text"].str.strip()
        data["text"] = data["text"].str.split(" ", expand=False)
        data["entity-label"] = data["entity-label"].str.split(",", expand=False)
        data = data.explode("text", ignore_index=True)
        data = data.dropna().reset_index().drop("index", axis=1)
        for index, row in data.iterrows():
        # in case a row (sentence) contains more than one entity-label pair
            for x in row["entity-label"]:
                if x == "": break
                entity, label = x.split(" - ")
                # the below condition has one flaw: eg. the word 'a' is in 'cat', so it
                # gets a label, but it is fixed in bio-tagging
                # TODO: will look again in this algorithm
                if row["text"].strip() in entity.strip():
                    data.loc[index, "tag"] = label.strip()
                    data.loc[index, "entity"] = entity.strip()
                    break
                else:
                    data.loc[index, "tag"] = "O"
                    data.loc[index, "entity"] = row["text"].strip()

        data = data[["Sentence #", "text","entity", "tag"]]
        data = data.dropna()
        data = bio_tag(data)

        if diacritics:
            data = handle_diacritics(data, task)

        if save:
            outpath = f"{task}-{file_name}.csv"
            data.to_csv(outpath, index=False)
        
        return data

    elif source_type == "doccano":
        data = pd.read_json(path_or_buf=file_path, lines=True)

        for index, row in data.iterrows():
            start = row.label[0][0]
            end = row.label[0][1]
            label = row.label[0][2]
            entity = row.text[start:end]
            data.loc[index, "entity"] = entity
            data.loc[index, "tag"] = label
            data.loc[index, "entity - label"] = [f"{entity} - {label}"]
            
        data = data.rename(columns={"id": "Sentence #"})
        data["Sentence #"] = "Sentence: " + data["Sentence #"].astype(str)
        data = data.drop(["label", "Comments"], axis=1)

        data["text"] = data["text"].str.split(" ", expand=False)
        data = data.explode("text", ignore_index=False)
        data = data.dropna().reset_index().drop("index", axis=1)

        for index, row in data.iterrows():
            text = data.loc[index, "text"].strip().translate(str.maketrans('', '', string.punctuation))
            if text not in data.loc[index, "entity"].strip():
                data.loc[index, "tag"] = "O"
        
        data = bio_tag(data)
        data = data[["Sentence #", "text", "tag"]]
        if diacritics:
            data = handle_diacritics(data, task)
        
        if save:
            outpath = f"{task}-{file_name}.csv"
            data.to_csv(outpath, index=False)

        return data
    else:
        sys.exit(f"Source: '{source_type}' is not supported")


def intents_args(args):
    """
    :param args: command line arguments
    """
    file_path = args.path
    separator = args.separator
    extention = args.extention
    file_name = args.out
    diacritics = True if args.diacritics=="yes" else False
    prepare_intent_dataset(file_path=file_path,
                           separator=separator,
                           extention=extention,
                           file_name=file_name,
                           task="intents",
                           diacritics=diacritics,
                           save=True)



def entities_args(args):
    """
    :param args: command line arguments
    """
    file_path = args.path
    separator = args.separator
    extention = args.extention
    file_name = args.out
    diacritics = True if args.diacritics=="yes" else False
    source_type = args.type
    prepare_entities_dataset(file_path=file_path,
                             separator=separator,
                             extention=extention,
                             source_type=source_type,
                             file_name=file_name,
                             task="entities",
                             diacritics=diacritics,
                             save=True)



def parse_args() -> argparse.Namespace:
    """
    reads all the command line arguments
    :return: Namespace with command line arguments
    """

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-p", "--path", type=str, required=True, help="data path")
    parent_parser.add_argument("-s", "--separator", type=str, required=True, help="delimiter: ',' / '\t'")
    parent_parser.add_argument("-e", "--extention", type=str, help="file extention: csv / json")
    parent_parser.add_argument("-o", "--out", type=str, help="final name for the produced file")
    parent_parser.add_argument("-d", "--diacritics", type=str, help="if 'yes', duplicate examples with no diacritcs", choices={"yes", "no"})

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    intents_parser = subparsers.add_parser("intents", help="create dataset for classification", aliases=["i"], parents=[parent_parser])
    intents_parser.set_defaults(func=intents_args)

    entities_parser = subparsers.add_parser("entities", help="create dataset for ner", aliases="e", parents=[parent_parser])
    entities_parser.add_argument("-t", "--type", type=str, help="type: manual / doccano")
    entities_parser.set_defaults(func=entities_args)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)

# python -m create-datasets intents -p data.csv -s ',' -e csv -o data -d yes
# python -m create-datasets entities -p data.csv -s ',' -e csv -o data -t manual
# python -m create-datasets entities -p admin.json -s ',' -e csv -o data -t doccano