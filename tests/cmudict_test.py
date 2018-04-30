import io
from tacotron.utils import cmudict


test_data = '''
;;; # CMUdict  --  Major Version: 0.07
)PAREN\tpɝˈɛn
'TIS\tˈtɪz
ADVERSE\tædˈvɝːs, ˈædˌvɝːs, ˌædˈvɝːs
ADVERSELY\tædˈvɝːsli
ADVERSITY\tædˈvɝːsɪˌtiː
BARBERSHOP\tˈbɑːrbɝˌʃɑːp
YOU'LL\tˈjuːl
'''


def test_cmudict():
  c = cmudict.CMUDict(io.StringIO(test_data))
  assert len(c) == 6
  assert len(cmudict.valid_symbols) == 41
  assert c.lookup('ADVERSITY') == ['ædˈvɝːsɪˌtiː']
  assert c.lookup('ADVERSITY', True)[0] == 'ædvɝsɪti'
  assert c.lookup('BarberShop') == ['ˈbɑːrbɝˌʃɑːp']
  assert c.lookup("You'll") == ['ˈjuːl']
  assert c.lookup("'tis") == ['ˈtɪz']
  assert c.lookup('adverse') == [
    'ædˈvɝːs',
    'ˈædˌvɝːs',
    'ˌædˈvɝːs'
  ]
  assert c.lookup('') == None
  assert c.lookup('', True) == None
  assert c.lookup('foo') == None
  assert c.lookup(')paren') == None


def test_cmudict_no_keep_ambiguous():
  c = cmudict.CMUDict(io.StringIO(test_data), keep_ambiguous=False)
  assert len(c) == 5
  assert c.lookup('adversity') == ['ædˈvɝːsɪˌtiː']
  assert c.lookup('adverse') == None
