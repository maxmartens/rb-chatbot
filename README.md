# Chatbot
Das ist ein sehr schöner Chatbot und sogar ein Markenchatbot ©

Es handelt sich um einen äußerst guten und kompetenten Chatbot,
basierend auf fortschrittlichsten Technologien aus dem Bereich 
der künstlichen Intelligenz.

## Debug Mode
Der Chatbot bietet eine Möglichkeit zur Ausgabe verschiedener Daten
im Programmablauf. Zur Aktivierung bzw. Deaktivierung muss folgende Zeile
in der Datei `chatbot.py` angepasst werden:
```
Logger(level=2, debug_mode=<Bool>)
```

## Modelldaten
Es werden vortrainierte Daten des Modells mit ausgeliefert. Sollte dies nicht
erwünscht sein und das Modell beim erstmaligen Start trainiert werden, so sind
folgende Dateien zu löschen:
* `model.tflearn.data-XXX-of-XXX`
* `model.tflearn.index`
* `model.tflearn.meta`
* `checkpoint`

Trotz solgfältiger Optimierung des Neuronalen Netzes kann es mit jedem Training
zu Abweichungen bei den Ergebnissen kommen.


## Bibliotheken
Zur Ausführung der Anwendung müssen folgende Bibliotheken installiert sein. 
Dabei ist zu beachten, dass die Bibliothek `libpostal` nicht mit Windows-Systemen
kompatibel ist.

* nltk
* numpy
* flearn
* tensorflow
* random
* json
* pickle
* pandas
* pandasql
* re
* libpostal (UNIX only)
* pypostal (UNIX only)

## Ausführung
Zur Ausführung der Anwendung müssen die angegebenen Bibliotheken installiert
und anschließend die Datei `chatbot.py` in der entsprechenden
Umgebung ausgeführt werden. Voraussetzung dafür ist Python 3.

Windows-Systeme sind mit der oben erwähnten Einschränkung kompatibel.
