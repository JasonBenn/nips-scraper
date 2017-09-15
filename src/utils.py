def ascii_alphafy(string):
  return string.replace(r':', '').replace(',', '').encode('ascii', errors='ignore')

def strip_text(string):
  return string.replace('\n', ' ').strip().encode("utf8")
