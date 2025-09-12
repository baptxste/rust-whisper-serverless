# WHISPER SERVERLESS

**Prérequis :**
 - télécharger le model whisper large v3 turbo depuis huggingface, dans le dossier ~/cache/huggingface/hub 

 https://huggingface.co/openai/whisper-large-v3-turbo



Selectionne le micro par défaut.

serverless -> le modèle est uniquement chargé en mémoire lors qu'une transcription est nécessaire. 

```bash
http POST http://0.0.0.0:3099/start  # pour démarrer la transcription
http POST http://0.0.0.0:3099/start  # pour stopper la transcription
http POST http://0.0.0.0:3099/status  # pour infos
```
## paramètres

--debug <true|false>        

--threshold <f32>             Seuil d'énergie pour déclencher la VAD (défaut : 0.3) *à ajuster en fonction du micro utilisé, le mode debug permet d'afficher le seuil d'énergie perçu*

--silence <u64>               Durée de silence en ms avant arrêt (défaut : 1500) *au bout de cette durée de silence la transcription est envoyée*

--min <u64>                   Durée minimale d'un segment valide en ms (défaut : 8000) *permet de limiter les envoies trop courts dûs à des bruits parasites*

--prebuffer <usize>          Taille du buffer audio avant la parole (défaut : 16000) *garde un buffer avant la détèction de la parole pour améliorer la transcription, si on ne fait pas cela le modèle échoue sur les premiers mots*

--endpoint_client            Adresse du client endpoint (défaut : None, affiche la transcrition en console)
