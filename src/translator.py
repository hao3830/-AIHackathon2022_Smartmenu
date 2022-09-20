from fuzzywuzzy import process
from pathlib import Path
from googletrans import Translator as GGTranslator
from .utils import load_reference_csv


class Translator:
    def __init__(self, referenced_csv=None):
        if referenced_csv:
            print("Translate refer CSV file")

            if isinstance(referenced_csv, str):
                referenced_csv = Path(referenced_csv)

            referenced_translate = load_reference_csv(referenced_csv, translate=True)

            self.referenced_translate = referenced_translate
            self.referenced_translate_keys = list(referenced_translate.keys())

        self.gg_translator = GGTranslator()

    def __call__(self, text):
        text = text.lower()
        if self.referenced_translate:
            if text in self.referenced_translate:
                return self.referenced_translate[text]

            key, percent = process.extractOne(text, self.referenced_translate_keys)
            if percent > 90:
                return self.referenced_translate[key]

            return self.gg_translator.translate(text, dest="en", src="vi").text

        return self.gg_translator.translate(text, dest="en", src="vi").text
