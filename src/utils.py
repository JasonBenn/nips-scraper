def clean_text(string):
  return string.replace(r':', '').replace(',', '').encode('ascii', errors='ignore')
