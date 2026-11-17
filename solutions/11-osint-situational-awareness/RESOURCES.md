# Solution 11 — Learning resources

## OSINT method and ethics

* Berkeley Protocol on Digital Open Source Investigations (UN OHCHR, method + ethics): <https://www.ohchr.org/en/publications/policy-and-methodological-publications/berkeley-protocol-digital-open-source>
* First Draft / Information Disorder resources (verification of public information): <https://firstdraftnews.org/>
* GDPR-style data minimisation principles (useful even where not legally required): <https://gdpr-info.eu/art-5-gdpr/>

## Data collection

* RSS 2.0 specification: <https://www.rssboard.org/rss-specification>
* feedparser documentation: <https://feedparser.readthedocs.io/>
* HTTP conditional requests (ETag, If-Modified-Since): <https://developer.mozilla.org/en-US/docs/Web/HTTP/Conditional_requests>
* robots.txt standard (RFC 9309): <https://www.rfc-editor.org/rfc/rfc9309>
* GDELT DOC 2.0 API: <https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/>

## Cyber threat intelligence

* CISA Known Exploited Vulnerabilities catalogue: <https://www.cisa.gov/known-exploited-vulnerabilities-catalog>
* Certificate Transparency explained: <https://certificate.transparency.dev/howctworks/>
* crt.sh: <https://crt.sh/>
* MITRE ATT&CK (to classify cyber events further): <https://attack.mitre.org/>

## NLP for Arabic and Kurdish news

* Masader (Arabic datasets, incl. news classification): <https://arbml.github.io/masader/>
* CAMeL Tools (Arabic morphology, normalisation): <https://github.com/CAMeL-Lab/camel_tools>
* KLPT Kurdish Language Processing Toolkit: <https://github.com/sinaahmadi/klpt>
* Granite 4.2 model card (structured output): <https://huggingface.co/ibm-granite/granite-4.2-8b>

## Search and storage

* SQLite FTS5: <https://www.sqlite.org/fts5.html>
* SQLite WAL mode: <https://www.sqlite.org/wal.html>

## IBM Power

* ppc64le/pyeco: <https://github.com/ppc64le/pyeco> · IBM wheel index: <https://wheels.developerfirst.ibm.com/ppc64le/linux>
* RocketCE conda channel: <https://anaconda.org/rocketce>
* IBM Power Virtual Server docs: <https://cloud.ibm.com/docs/power-iaas>

## Practice exercises

1. Add a source type for an official JSON API you are allowed to use (e.g. a weather service) and map it to events.
2. Draw the day's events on a governorate map (GeoJSON of Iraq's governorates + Leaflet).
3. Add a "story timeline" API endpoint that returns all events of one story in order.
4. Compare word-set clustering with Granite Embedding 278M multilingual clustering on 200 labelled pairs.
