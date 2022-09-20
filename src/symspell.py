from symspellpy import SymSpell


class SymSpellWrapper:
    def __init__(self, dictionary_path="src/symspell_dictionary.txt"):
        sym_spell = SymSpell()
        sym_spell.load_dictionary(dictionary_path, 0, 1, encoding="utf=8")
        self.sym_spell = sym_spell

    def correct(self, input_terms, max_edit_distance=2):
        input_terms = input_terms.lower()
        input_len = len(input_terms.split())
        suggestions = self.sym_spell.lookup_compound(
            input_terms, max_edit_distance=max_edit_distance
        )

        if len(suggestions) == 0:
            return input_terms

        for suggest in suggestions:
            term = suggest.term
            dist = suggest.distance
            if dist < input_len:
                return term

        return input_terms
