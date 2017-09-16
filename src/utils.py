# -*- coding: utf-8 -*-

def ascii_alphafy(string):
  return string.replace(r':', '').replace(',', '').decode('utf8', 'ignore').encode('ascii', errors='ignore')

def strip_text(string):
  return string.replace('\n', ' ').strip().encode("utf8")

class RateLimitError(ValueError):
  """For Google rate limits"""
