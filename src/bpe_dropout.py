import random
from typing import Any, Dict, List, Optional

import sentencepiece as spm
from transformers import RobertaTokenizer, XLMRobertaTokenizer


def get_pairs(word, alpha=0):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        if alpha < random.random() or prev_char == "Ġ":
            pairs.add((prev_char, char))
        else:
            pairs.add((prev_char, "@@" + char))
        prev_char = char
    return pairs


class RobertaTokenizerDropout(RobertaTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        alpha=0,
        seed=42,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self.alpha = alpha
        self._alpha = alpha
        self.seed = seed
        random.seed(self.seed)

    def const_tokenize(self):
        self.alpha = 0

    def random_tokenize(self):
        self.alpha = self._alpha

    def reset_seed(self):
        random.seed(self.seed)

    def bpe(self, token):
        if (token in self.cache) and (self.alpha == 0):
            # サブワード正則化時はキャッシュから返さない
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word, self.alpha)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word, self.alpha)
        word = " ".join(word)
        if self.alpha == 0:
            # サブワード正則化時の分割はキャッシュしない
            self.cache[token] = word
        return word

    def tokenzizeSentence(self, text):
        text = text.split(" ")
        subwords = []
        word_ids = [-1]
        for i, token in enumerate(text):
            if i == 0:
                subword = self.tokenize(token)
            else:
                subword = self.tokenize(" " + token)
            subwords += subword
            word_ids += [max(word_ids) + 1] * len(subword)

        return subwords, word_ids[1:]


class XLMRobertaTokenizerDropout(XLMRobertaTokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        seed=42,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=sp_model_kwargs,
            **kwargs,
        )
        self.seed = seed
        random.seed(self.seed)
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.sp_model_kwargs["enable_sampling"] = False
        self.state_sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.state_sp_model.Load(str(vocab_file))
        self.using_sp_model = self.sp_model

    def _tokenize(self, text: str) -> List[str]:
        # TODO check if the t5/llama PR also applies here
        return self.using_sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        if hasattr(self, "using_sp_model"):
            spm_id = self.using_sp_model.PieceToId(token)
        else:
            spm_id = self.sp_model.PieceToId(token)
        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        if hasattr(self, "using_sp_model"):
            tokens = self.using_sp_model.IdToPiece(index - self.fairseq_offset)
        else:
            tokens = self.sp_model.IdToPiece(index - self.fairseq_offset)
        return tokens

    def tokenzizeSentence(self, text):
        text = text.split(" ")
        subwords = []
        word_ids = [-1]
        for i, token in enumerate(text):
            if i == 0:
                subword = self.tokenize(token)
            else:
                subword = self.tokenize(" " + token)
            subwords += subword
            word_ids += [max(word_ids) + 1] * len(subword)

        return subwords, word_ids[1:]

    def const_tokenize(self):
        self.using_sp_model = self.state_sp_model

    def random_tokenize(self):
        self.using_sp_model = self.sp_model

    def reset_seed(self):
        random.seed(self.seed)


if __name__ == "__main__":
    tokenizer: RobertaTokenizerDropout = RobertaTokenizerDropout.from_pretrained("roberta-base", alpha=0.2)
    text = "This is a rtd pen."

    print("random tokenize")
    for i in range(10):
        a = tokenizer.tokenize(text)
        print(a)

    print("random tokenize")
    tokenizer.reset_seed()
    for i in range(10):
        a = tokenizer.tokenize(text)
        print(a)

    tokenizer.const_tokenize()
    print("const tokenize")
    for i in range(10):
        a = tokenizer.tokenize(text)
        print(a)
