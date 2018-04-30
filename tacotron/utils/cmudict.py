import re

valid_symbols = [' ', ',', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
				 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'z',
				 'æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɪ', 'ʃ', 'ʊ', 'ʌ',
				 'ʒ', 'ˈ', 'ˌ', 'ː', 'θ']

_valid_symbol_set = set(valid_symbols)
_emphasis_symbols = r"[ˈ,ˌː]"


class CMUDict:
	'''
	Thin wrapper around CMUDict data converted to IPA.
	See http://www.speech.cs.cmu.edu/cgi-bin/cmudict
	and https://github.com/menelik3/cmudict-ipa
	'''
	def __init__(self, file_or_path, keep_ambiguous=True):
		if isinstance(file_or_path, str):
			with open(file_or_path) as f:
				entries = _parse_cmudict(f)
		else:
			entries = _parse_cmudict(file_or_path)
		if not keep_ambiguous:
			entries = {word: pron for word, pron in entries.items()
					   if len(pron) == 1}
		self._entries = entries


	def __len__(self):
		return len(self._entries)


	def lookup(self, word, strip_emphasis=False):
		'''Returns a list of IPA pronunciations of the given word.'''
		pronunciations = self._entries.get(word.upper())
		if pronunciations and strip_emphasis:
			pronunciations = [self.strip_emphasis(p) for p in pronunciations]
		return pronunciations

	def strip_emphasis(self, pronunciation):
		return re.sub(_emphasis_symbols, '', pronunciation)



def _parse_cmudict(file):
	cmudict = {}
	for line in file:
		if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
			parts = line.split('\t')
			if len(parts) > 1:
				cmudict[parts[0]] = [p for p in _get_pronunciations(parts[1])
									 if p]
	return cmudict


def _get_pronunciations(s):
	prons = s.strip().split(', ')
	return [_validate_pronunciation(p) for p in prons]


def _validate_pronunciation(pronunciation):
	for char in pronunciation:
		if char not in _valid_symbol_set:
			return None
	return pronunciation
