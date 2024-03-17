from urllib.request import urlopen
import os
import ssl
import platform

url = "https://www.cosmopolitan.com/uk/entertainment/a29755553/harry-potter-quiz/"
page = urlopen(url)
print(page)

html_bytes = page.read()
html = html_bytes.decode("utf-8")

print(html.find("<li>"))
print(html[81260:81364])

